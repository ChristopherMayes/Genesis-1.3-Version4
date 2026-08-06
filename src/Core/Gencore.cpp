#include "Gencore.h"

#ifdef USE_DPI
  #include "DiagnosticHookS.h"
#endif

#include "GPUEngine.h"

#include <chrono>
#include <memory>

extern bool MPISingle;

bool Gencore::run(Beam *beam, vector<Field*> *field, Setup *setup, Undulator *und,bool isTime, bool isScan, bool periodic, FilterDiagnostics &filter, bool gpu, bool gpuValidate)
{
    // function returns 'true' if everything is ok


    //-------------------------------------------------------
    // init MPI and get size etc.
    //
    int size=1;
    int rank=0;
	if (!MPISingle){
	    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // assign rank to node
	    MPI_Comm_size(MPI_COMM_WORLD, &size); // assign rank to node
    }

    if (rank==0) {
        cout << endl << "Running Core Simulation..." << endl;
    }

    //-----------------------------------------
	// init beam, field and undulator class

    string rn, fnbase;
    setup->getRootName(&rn);
    setup->RootName_to_FileName(&fnbase, &rn); // includes .RunX. if not the first &track command
    Control   *control=new Control;
    control->init(rank,size,fnbase,beam,field,und,isTime,isScan, periodic);

    Diagnostic diag;
#ifdef USE_DPI
    und->plugin_info_txt.clear();

    for(int kk=0; kk<setup->diagpluginfield_.size(); kk++) {
	if(rank==0) {
            cout << "Setting up DiagFieldHook for libfile=\"" << setup->diagpluginfield_.at(kk).libfile << "\", obj_prefix=\"" << setup->diagpluginfield_.at(kk).obj_prefix << "\"" << endl;
        }
        DiagFieldHook *pdfh = new DiagFieldHook(); /* !do not delete this instance, it will be destroyed when DiagFieldHook instance is deleted! */
        bool diaghook_ok = pdfh->init(&setup->diagpluginfield_.at(kk));
        if(diaghook_ok) {
	    pdfh->set_runid(setup->getCount()); // propagate run id so that it can be used in the plugins, for instance for filename generation

	    string tmp_infotxt = pdfh->get_info_txt();
	    und->plugin_info_txt.push_back(tmp_infotxt);
	    stringstream tmp_prefix;
	    tmp_prefix << "/Field/" << setup->diagpluginfield_.at(kk).obj_prefix;
	    und->plugin_hdf5_prefix.push_back(tmp_prefix.str());

            diag.add_field_diag(pdfh);
            if(rank==0) {
                cout << "DONE: Registered DiagFieldHook" << endl;
            }
        } else {
            delete pdfh;
            if(rank==0) {
                cout << "failed to set up DiagFieldHook, not registering" << endl;
            }
        }
    }

	for(int kk=0; kk<setup->diagpluginbeam_.size(); kk++) {
		if(rank==0) {
			cout << "Setting up DiagBeamHook for libfile=\"" << setup->diagpluginbeam_.at(kk).libfile << "\", obj_prefix=\"" << setup->diagpluginbeam_.at(kk).obj_prefix << "\"" << endl;
		}
		DiagBeamHook *pdbh = new DiagBeamHook(); /* !do not delete this instance, it will be destroyed when DiagBeamHook instance is deleted! */
		bool diaghook_ok = pdbh->init(&setup->diagpluginbeam_.at(kk));
		if(diaghook_ok) {
			pdbh->set_runid(setup->getCount()); // propagate run id so that it can be used in the plugins, for instance for filename generation

			string tmp_infotxt = pdbh->get_info_txt();
			und->plugin_info_txt.push_back(tmp_infotxt);
			stringstream tmp_prefix;
			tmp_prefix << "/Beam/" << setup->diagpluginbeam_.at(kk).obj_prefix;
			und->plugin_hdf5_prefix.push_back(tmp_prefix.str());

			diag.add_beam_diag(pdbh);
			if(rank==0) {
				cout << "DONE: Registered DiagBeamHook" << endl;
			}
		} else {
			delete pdbh;
			if(rank==0) {
				cout << "failed to set up DiagBeamHook, not registering" << endl;
			}
		}
	}
#endif
    if (rank==0) { cout << "Initial analysis of electron beam and radiation field..."  << endl; }
    diag.init(rank, size, und->outlength(), beam->beam.size(),field->size(),isTime,isScan,filter);
    diag.calc(beam, field, und->getz());  // initial calculation

    // Non-null only when an accelerator backend owns the field, so that the
    // slippage can keep the backend's copy in step one slice at a time.
    SliceSync *slipSync = nullptr;

    // GPU backend, selected with `gpu = true` in &track. `gpu_validate = true`
    // additionally runs the CPU path every step and reports the largest
    // relative difference; that is a testing mode and is much slower than
    // either path on its own.
    //
    // Everything below is written against GPUEngine, so this file contains
    // nothing device specific and compiles unchanged in a build with no
    // backend, where create() reports that and the run stops.
    unique_ptr<GPUEngine> engine;
    const bool useGPU = gpu || gpuValidate;
    const bool gpuDrive = gpu;             // GPU result is the answer
    const bool gpuCompare = gpuValidate;   // also run the CPU and diff
    double gpuFieldErr = 0;
    double gpuBeamErr = 0;
    bool gpuBeamOK = false;
    int gpuSteps = 0;
    int gpuFallback = 0;
    BeamSliceMoments gpuBM;
    vector<FieldSliceMoments> gpuFM;
    auto gpuMoments = [&](vector<Field *> *f) -> bool {
        if (!engine->beamMoments(filter.beam.harm, filter.beam.auxiliar, gpuBM)) {
            return false;
        }
        gpuFM.resize(f->size());
        for (size_t i = 0; i < f->size(); i++) {
            if (!engine->fieldMoments(static_cast<int>(i), filter.field.fft, gpuFM[i])) {
                return false;
            }
        }
        return true;
    };

    // Lets Control::applySlippage move its one slice per slip event without
    // dragging the whole field across.
    struct EngineSliceSync : public SliceSync {
        GPUEngine *e {nullptr};
        vector<Field *> *f {nullptr};
        void pullSlice(int ifld, int is) override { e->downloadFieldSlice(ifld, is, f->at(ifld)); }
        void pushSlice(int ifld, int is) override { e->uploadFieldSlice(ifld, is, f->at(ifld)); }
    } gpuSync;
    gpuSync.f = field;
    if (useGPU) {
        string reason;
        engine.reset(GPUEngine::create(reason));
        if (engine == nullptr) {
            // reason set by create()
        } else if (!engine->init(beam, field, reason)) {
            // reason set by init()
        } else if (beam->gpuUnsupportedPhysics(reason)) {
            reason += " is not implemented on the GPU yet";
        } else {
            reason.clear();
        }
        if (!reason.empty()) {
            // The user asked for the GPU explicitly, so do not quietly fall
            // back to the CPU and hand back numbers from a different code path.
            if (rank == 0) {
                cout << "*** Error: gpu = true in &track, but " << reason << endl;
            }
            return false;
        }
        gpuSync.e = engine.get();
        engine->upload(beam, field);
        // In drive mode the host arrays are no longer the source of truth, so
        // the slippage syncs itself one slice at a time and nothing else copies
        // per step. In compare mode the CPU path runs too and both copies have
        // to be updated the plain way.
        if (gpuDrive && !gpuCompare) { slipSync = &gpuSync; }
        GPUEngine::SyncError err = engine->compare(beam, field);
        if (rank == 0) {
            cout << GPUEngine::backend() << " backend: " << engine->deviceName() << ", "
                 << engine->bytesResident() / (1024 * 1024) << " MB resident, gamma_ref = "
                 << engine->gammaRef() << endl;
            cout << "  host transfer check: field " << err.field
                 << "   beam " << err.beam << " (relative, FP32 rounding is ~1e-7)" << endl;
            if (gpuCompare) {
                cout << "  gpu_validate is on: the CPU path runs as well and the "
                        "difference is reported. This is slow." << endl;
            }
        }
    }

    // Wall clock of the tracking loop, for the GPU report at the end of it.
    // Genesis' own figure is clock(), which counts neither the wait on the
    // device nor the shader compile, which runs in another process.
    const auto loopStart = chrono::steady_clock::now();

	/*************/
	/* MAIN LOOP */
	/*************/
	while(und->advance(rank))
	{
	  double delz=und->steplength();

	  // ----------------------------------------
	  // step 1 - apply most marker action  (always at beginning of a step)
	  bool error_IO=false;
	  // A dump reads the host arrays, so the resident copy has to come back
	  // first. getMarker() reports what is due before applyMarker acts on it:
	  // bit 1 field dump, bit 2 beam dump, bit 4 sort.
	  //
	  // Nothing goes back afterwards. The dump is read-only, and Beam::sort()
	  // only does anything for one4one, which the backend refuses. An upload
	  // here would be actively wrong in any case: this state is from the top of
	  // the step, and by the time step 3 runs the GPU has already advanced it,
	  // so writing it back rolls the beam back one step.
	  if (gpuDrive && !gpuCompare) {
	    const int mk = und->getMarker();
	    if ((mk & 1) != 0) { engine->downloadField(field); }
	    if ((mk & 2) != 0) { engine->downloadBeam(beam); }
	  }
	  bool sort=control->applyMarker(beam, field, und, error_IO);
	  if(error_IO) {
	    return(false);
	  }


	  // ---------------------------------------
	  // step 2 - Advance electron beam

	  if (useGPU) {
	    string why;
	    // In drive mode the GPU copy is already the current state; the upload
	    // is only there to feed the CPU path that compare mode also runs.
	    if (!gpuDrive || gpuCompare) { engine->upload(beam, field); }
	    gpuBeamOK = engine->beamStep(beam, und, field, delz, why);
	    if (!gpuBeamOK) {
	      // Nothing triggers this today, but the path is kept so that an
	      // element ported later can be refused rather than dropped.
	      // Falling back needs the host arrays, so bring them over.
	      if (gpuDrive && !gpuCompare) { engine->download(beam, field); }
	      if (gpuFallback == 0 && rank == 0) {
		cout << "  GPU: falling back to the CPU for " << why << " steps" << endl;
	      }
	      gpuFallback++;
	    }
	  }
	  if (!gpuDrive || !gpuBeamOK || gpuCompare) {
	    beam->track(delz,field,und);
	  }
	  if (useGPU) {
	    if (!gpuBeamOK) {
	      // The GPU refused this step and the CPU took it instead, so the host
	      // now holds the answer. There is nothing to compare, because the only
	      // difference measured would be the step the GPU did not take, and
	      // nothing to copy back, because doing so would undo the step.
	      engine->upload(beam, field);
	    } else {
	      if (gpuCompare) {
	        GPUEngine::SyncError e = engine->compare(beam, field);
	        gpuBeamErr = max(gpuBeamErr, e.beam);
	      }
	      if (gpuDrive && gpuCompare) { engine->downloadBeam(beam); }
	    }
	  }

	  // -----------------------------------------
	  // step 3 - Beam post processing, e.g. sorting


	  if (sort){
	    int shift=beam->sort();

	    if (shift!=0){
	      for (int i=0;i<field->size();i++){
		      control->applySlippage(shift, field->at(i), i, slipSync);
	      }
	    }
	  }
  
	  // ---------------------------------------
	  // step 4 - Advance radiation field

	  if (useGPU) {
	    if (!gpuDrive || gpuCompare) { engine->upload(beam, field); }
	    engine->fieldStep(und, field, delz);
	  }
	  if (!gpuDrive || gpuCompare) {
	    for (int i=0; i<field->size();i++){
	      field->at(i)->track(delz,beam,und);
	    }
	  }
	  if (useGPU) {
	    if (gpuCompare) {
	      GPUEngine::SyncError e = engine->compare(beam, field);
	      gpuFieldErr = max(gpuFieldErr, e.field);
	    }
	    gpuSteps++;
	    if (gpuDrive && gpuCompare) { engine->downloadField(field); }
	  }


	  //-----------------------------------------
	  // step 5 - Apply slippage

	  for (int i=0;i<field->size();i++){
	    control->applySlippage(und->slippage(), field->at(i), i, slipSync);
	  }

	  //-------------------------------
	  // step 6 - Calculate beam parameter stored into a buffer for output

	  //beam->diagnostics(und->outstep(),und->getz());
	  //for (int i=0;i<field->size();i++){
	  //  field->at(i)->diagnostics(und->outstep());
	  //}

	  if (und->outstep()) {
	    // The diagnostics are a per-slice reduction over exactly the arrays
	    // that already live on the GPU, and they dominate the run once the
	    // tracking is fast: on 500 slices at ngrid=256 they were 76% of the
	    // wall time. Reduce them there instead of on the host.
	    if (gpuDrive && gpuMoments(field)) {
	      diag.calc(beam, field, und->getz(), &gpuBM, &gpuFM);
	    } else {
	      diag.calc(beam, field, und->getz());
	    }
	  }
	}

	if (useGPU) {
	  const double loopSec =
	      chrono::duration<double>(chrono::steady_clock::now() - loopStart).count();
	  // The loop kept the state on the GPU; the host owns it again from here,
	  // for the closing marker action, the output file and whatever namelist
	  // follows this &track.
	  if (gpuDrive && !gpuCompare) { engine->download(beam, field); }
	  if (rank == 0) {
	    // Device busy time against the wall clock of the loop. A device that is
	    // already saturated will not go faster with more MPI ranks pointed at
	    // it; one that is idling says the host is the limit.
	    const double busy = engine->deviceSeconds();
	    cout << GPUEngine::backend() << ": " << gpuSteps << " steps in " << loopSec
		 << " s, device busy " << busy << " s";
	    if (loopSec > 0) {
	      cout << " (" << static_cast<int>(100.0 * busy / loopSec + 0.5) << "%)";
	    }
	    cout << endl;
	    if (gpuCompare) {
	      cout << GPUEngine::backend() << " vs CPU over " << gpuSteps
		   << " steps: max relative error, field "
		   << gpuFieldErr << ", beam " << gpuBeamErr << endl;
	    }
	    if (gpuFallback > 0) {
	      cout << GPUEngine::backend() << ": " << gpuFallback << " of " << gpuSteps
		   << " steps fell back to the CPU" << endl;
	    }
	  }
	}
     
        //---------------------------
        // end and clean-up 

	// perform last marker action
        bool error_IO=false;
	bool sort=control->applyMarker(beam, field, und, error_IO);
	if(error_IO) {
	  return(false);
	}
	if (sort){
	    int shift=beam->sort();

	    if (shift!=0){
	      for (int i=0;i<field->size();i++){
		    control->applySlippage(shift, field->at(i));
	      }
	    }
	}


	/* write out diagnostic arrays */
	if (rank==0){
	  cout << "Writing output file..." << endl;
	}

	// control->output(beam,field,und,diag);
	if(!diag.writeToOutputFile(beam, field, setup, und)) {
	  delete control;
	  return(false);
	}

	delete control;
      
    if (rank==0){
	  cout << endl << "Core Simulation done." << endl;
    }
    return(true);
}
