#include "Gencore.h"

#ifdef USE_DPI
  #include "DiagnosticHookS.h"
#endif

#ifdef G4_METAL
  #include "MetalEngine.h"
  #include <cstdlib>
#endif

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

#ifdef G4_METAL
    // GPU backend, selected with `gpu = true` in &track. `gpu_validate = true`
    // additionally runs the CPU path every step and reports the largest
    // relative difference; that is a testing mode and is much slower than
    // either path on its own.
    MetalEngine metal;
    bool useMetal = gpu || gpuValidate;
    bool metalDrive = gpu;              // GPU result is the answer
    bool metalCompare = gpuValidate;    // also run the CPU and diff
    double metalFieldErr = 0;
    double metalBeamErr = 0;
    bool metalBeamOK = false;
    int metalSteps = 0;
    int metalFallback = 0;
    if (useMetal) {
        string reason;
        if (!MetalEngine::available()) {
            reason = "no Metal device with unified memory";
        } else if (!metal.init(beam, field, reason)) {
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
        metal.upload(beam, field);
        MetalEngine::SyncError err = metal.compare(beam, field);
        if (rank == 0) {
            cout << "Metal backend: " << MetalEngine::deviceName() << ", "
                 << metal.bytesResident() / (1024 * 1024) << " MB resident, gamma_ref = "
                 << metal.gammaRef() << endl;
            cout << "  host transfer check: field " << err.field
                 << "   beam " << err.beam << " (relative, FP32 rounding is ~1e-7)" << endl;
            if (metalCompare) {
                cout << "  gpu_validate is on: the CPU path runs as well and the "
                        "difference is reported. This is slow." << endl;
            }
        }
    }
#endif

	/*************/
	/* MAIN LOOP */
	/*************/
	while(und->advance(rank))
	{
	  double delz=und->steplength();

	  // ----------------------------------------
	  // step 1 - apply most marker action  (always at beginning of a step)
	  bool error_IO=false;
	  bool sort=control->applyMarker(beam, field, und, error_IO);
	  if(error_IO) {
	    return(false);
	  }


	  // ---------------------------------------
	  // step 2 - Advance electron beam

#ifdef G4_METAL
	  // The upload is still needed every step because the host arrays remain the
	  // source of truth for diagnostics, slippage and sorting. Removing it is the
	  // next stage of the port.
	  if (useMetal) {
	    string why;
	    metal.upload(beam, field);
	    metalBeamOK = metal.beamStep(beam, und, field, delz, why);
	    if (!metalBeamOK) {
	      // Localised elements the GPU does not handle (chicane, corrector).
	      // Falling back for that step is correct because the host arrays are
	      // in sync at this point.
	      if (metalFallback == 0 && rank == 0) {
		cout << "  Metal: falling back to the CPU for " << why << " steps" << endl;
	      }
	      metalFallback++;
	    }
	  }
	  if (!metalDrive || !metalBeamOK || metalCompare) {
	    beam->track(delz,field,und);
	  }
	  if (useMetal && metalBeamOK) {
	    if (metalCompare) {
	      MetalEngine::SyncError e = metal.compare(beam, field);
	      metalBeamErr = max(metalBeamErr, e.beam);
	    }
	    if (metalDrive) { metal.downloadBeam(beam); }
	  }
#else
	  beam->track(delz,field,und);
#endif

	  // -----------------------------------------
	  // step 3 - Beam post processing, e.g. sorting


	  if (sort){
	    int shift=beam->sort();

	    if (shift!=0){
	      for (int i=0;i<field->size();i++){
		      control->applySlippage(shift, field->at(i));
	      }
	    }
	  }
  
	  // ---------------------------------------
	  // step 4 - Advance radiation field

#ifdef G4_METAL
	  if (useMetal) {
	    metal.upload(beam, field);
	    metal.fieldStep(und, field, delz);
	  }
	  if (!metalDrive || metalCompare) {
	    for (int i=0; i<field->size();i++){
	      field->at(i)->track(delz,beam,und);
	    }
	  }
	  if (useMetal) {
	    if (metalCompare) {
	      MetalEngine::SyncError e = metal.compare(beam, field);
	      metalFieldErr = max(metalFieldErr, e.field);
	    }
	    metalSteps++;
	    if (metalDrive) { metal.downloadField(field); }
	  }
#else
	  for (int i=0; i<field->size();i++){
	    field->at(i)->track(delz,beam,und);
	  }
#endif


	  //-----------------------------------------
	  // step 5 - Apply slippage

	  for (int i=0;i<field->size();i++){
	    control->applySlippage(und->slippage(), field->at(i));  
	  }

	  //-------------------------------
	  // step 6 - Calculate beam parameter stored into a buffer for output

	  //beam->diagnostics(und->outstep(),und->getz());
	  //for (int i=0;i<field->size();i++){
	  //  field->at(i)->diagnostics(und->outstep());
	  //}

	  if (und->outstep()) {
	    diag.calc(beam, field, und->getz());
	  }
	}

#ifdef G4_METAL
	if (useMetal && rank == 0) {
	  if (metalCompare) {
	    cout << "Metal vs CPU over " << metalSteps << " steps: max relative error, field "
		 << metalFieldErr << ", beam " << metalBeamErr << endl;
	  }
	  if (metalFallback > 0) {
	    cout << "Metal: " << metalFallback << " of " << metalSteps
		 << " steps fell back to the CPU" << endl;
	  }
	}
#endif
     
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
