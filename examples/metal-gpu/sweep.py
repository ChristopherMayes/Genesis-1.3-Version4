#!/usr/bin/env python3
"""Sweep the Metal GPU backend against the CPU across a matrix of deck options.

    export FI_PROVIDER=tcp
    python3 sweep.py --genesis ../../build-metal/genesis4

Every case is generated from one base deck, so a failure names the single
option that caused it.

There are two tiers, because they catch different mistakes.

  step   `gpu_validate = true`. Both paths advance one step from the same
         state, the difference is measured, and the GPU result is then copied
         back to the host. Divergence cannot accumulate, so the number
         reported is the error of a single step and a wrong kernel shows up
         as an unmistakable jump rather than as chaotic growth. This is the
         cheap tier and most cases live here.

  run    Two separate runs, one CPU and one GPU, compared through the output
         file. The step tier cannot see residency bugs, because it copies the
         GPU state back to the host every step and so repairs anything that
         should have been transferred and was not. Only a full run with the
         host arrays left alone exercises that path.

         An end-to-end difference cannot be judged against a fixed tolerance,
         because it depends on how many steps the deck takes and on how much
         round-off each one contributes. Nor can it be judged against a small
         perturbation of the input: scaling the seed power by one part in 1e7
         moves the answer by one part in 1e6, because a uniform scale is very
         nearly an eigenmode of the amplifier and so is barely amplified.

         What the difference is judged against instead is the step tier's own
         measurement of the same deck. Each case is run a third time under
         gpu_validate, which reports the largest single-step difference and the
         number of steps. Their product is the difference that would result if
         every step's round-off added coherently, which is the worst that pure
         round-off can do. A run that stays under it has accumulated no faster
         than its own arithmetic error, whereas a missed transfer is a finite
         error rather than a round-off one and goes straight through the bound.

A case may also expect to be refused. Physics the GPU does not implement must
produce a hard error naming itself, never a quiet fallback that returns
numbers from a different code path.

Needs numpy and h5py for the run tier only.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# The base deck. Steady state, seeded, shot noise off, so that a case is
# reproducible and any difference is the GPU rather than the loading.
# --------------------------------------------------------------------------

BASE = [
    ("setup", {
        "rootname": "CASE",
        "lattice": "CASE.lat",
        "beamline": "ARAMIS",
        "lambda0": 1e-10,
        "gamma0": 11357.82,
        "delz": 0.045,
        "shotnoise": False,
        "nbins": 8,
        "npart": 8192,
        "beam_global_stat": True,
        "field_global_stat": True,
    }),
    ("lattice", {"zmatch": 9.5}),
    ("field", {"power": 5e3, "dgrid": 2.0e-4, "ngrid": 256, "waist_size": 30e-6}),
    ("beam", {"current": 3000, "delgam": 1.0, "ex": 4.0e-7, "ey": 4.0e-7}),
    ("track", {
        "gpu": True,
        "gpu_validate": True,
        "fft_fieldsolver": True,
        "bunchharm": 1,
        "zstop": 10,
    }),
]

LAT = """\
D1: DRIFT = {{ l = 0.44 }};
D2: DRIFT = {{ l = 0.24 }};
QF: QUADRUPOLE = {{ l = 0.08, k1 = {k1} }};
QD: QUADRUPOLE = {{ l = 0.08, k1 = -{k1} }};
UND: UNDULATOR = {{ lambdau = 0.015, nwig = 266, aw = 0.84853, helical = {hel}{extra} }};
FODO: LINE = {{ UND, D1, QF, D2, UND, D1, QD, D2 }};
ARAMIS: LINE = {{ 6*FODO }};
"""


def lattice(hel=True, k1=2.0, extra="", quad=""):
    text = LAT.format(hel="True" if hel else "False", k1=k1, extra=extra)
    if quad:
        text = text.replace("l = 0.08,", "l = 0.08, " + quad + ",")
    return text


# A taper, written as six undulator modules of decreasing strength rather than
# through a namelist, so that nothing but the lattice differs from the base.
LAT_TAPER = "".join(
    "U%d: UNDULATOR = { lambdau = 0.015, nwig = 266, aw = %.6f, helical = True };\n"
    % (i, 0.84853 * (1.0 - 0.0015 * i)) for i in range(12)
) + """\
D1: DRIFT = { l = 0.44 };
D2: DRIFT = { l = 0.24 };
QF: QUADRUPOLE = { l = 0.08, k1 = 2.0 };
QD: QUADRUPOLE = { l = 0.08, k1 = -2.0 };
ARAMIS: LINE = { U0,D1,QF,D2,U1,D1,QD,D2, U2,D1,QF,D2,U3,D1,QD,D2,
                 U4,D1,QF,D2,U5,D1,QD,D2, U6,D1,QF,D2,U7,D1,QD,D2,
                 U8,D1,QF,D2,U9,D1,QD,D2, U10,D1,QF,D2,U11,D1,QD,D2 };
"""

# A chicane and a corrector both take the place of a drift, so the lattice
# length is unchanged and the case stays comparable with the others.
LAT_CHICANE = LAT.format(hel="True", k1=2.0, extra="").replace(
    "D2: DRIFT = { l = 0.24 };",
    "D2: DRIFT = { l = 0.24 };\n"
    "CH: CHICANE = { l = 0.24, lb = 0.05, ld = 0.04, delay = 2e-6 };",
).replace("UND, D1, QF, D2, UND, D1, QD, D2", "UND, D1, QF, CH, UND, D1, QD, D2")

LAT_CORRECTOR = LAT.format(hel="True", k1=2.0, extra="").replace(
    "D2: DRIFT = { l = 0.24 };",
    "D2: DRIFT = { l = 0.24 };\nCO: CORRECTOR = { l = 0.24, cx = 2e-6, cy = -1e-6 };",
).replace("UND, D1, QF, D2, UND, D1, QD, D2", "UND, D1, QF, CO, UND, D1, QD, D2")

# A phase shifter, taking part of the drift so the lattice length is unchanged.
LAT_PHASESHIFTER = LAT.format(hel="True", k1=2.0, extra="").replace(
    "D1: DRIFT = { l = 0.44 };",
    "D1: DRIFT = { l = 0.34 };\nPS: PHASESHIFTER = { l = 0.1, phi = 1.9 };",
).replace("UND, D1, QF", "UND, PS, D1, QF")

TIME = "&time\ns0 = 0\nslen = 2e-8\nsample = 1\n&end\n"


class Case:
    def __init__(self, name, tier="step", edit=None, pre="", extra="", tail="",
                 lat=None, ranks=1, expect="ok", note=""):
        self.name = name
        self.tier = tier
        self.edit = edit or {}
        self.pre = pre              # namelists inserted before &field
        self.extra = extra          # namelists inserted before &track
        self.tail = tail            # namelists appended after &track
        self.lat = lat if lat is not None else lattice()
        self.ranks = ranks
        self.expect = expect        # ok | fallback | error:<substring>
        self.note = note


def td(edit=None, **kw):
    """A time-dependent case: shot noise on, a time window, four ranks.

    &time has to come before &field and &beam, because it sets the window they
    are generated over. Put it after them and the run is silently steady state.
    """
    e = {"setup.shotnoise": True}
    e.update(edit or {})
    kw.setdefault("ranks", 4)
    kw["pre"] = TIME + kw.get("pre", "")
    return e, kw


CASES = []


def case(*a, **kw):
    CASES.append(Case(*a, **kw))


# -- the undulator itself ---------------------------------------------------
case("helical", note="the reference point")
case("planar", lat=lattice(hel=False))
case("planar_split_focus", lat=lattice(hel=False, extra=", kx = 0.5, ky = 0.5"))
case("undulator_offset", lat=lattice(extra=", ax = 2e-5, ay = -1.5e-5"))
case("undulator_gradient", lat=lattice(extra=", gradx = 0.5, grady = -0.3"))
case("undulator_rolloff", lat=lattice(extra=", kx = 0.2, ky = 0.8"))
case("taper", lat=LAT_TAPER)
case("phaseshifter", lat=LAT_PHASESHIFTER)

# -- the transport ----------------------------------------------------------
case("quad_offset", lat=lattice(quad="dx = 5e-5, dy = -5e-5"))
case("quad_strong", lat=lattice(k1=6.0))
case("quad_weak", lat=lattice(k1=0.8))

# -- undulator errors -------------------------------------------------------
case("field_error", lat=lattice(hel=False),
     edit={"lattice.fielderror": 0.01, "lattice.seed": 84621})
case("orbit_error", lat=lattice(hel=False),
     edit={"lattice.fielderror": 0.01, "lattice.orbiterror": True,
           "lattice.seed": 84621})

# -- the grid ---------------------------------------------------------------
# The particle count rises with the grid. A bilinear deposition of a fixed
# number of particles onto a finer mesh puts fewer of them in each cell, and the
# shot noise that leaves in the source term is high spatial frequency, which is
# exactly where single precision is weakest. Measured at ngrid = 1024: 8.5e-03
# at npart = 8192, 3.2e-03 at 32768, 1.2e-03 at 131072. Holding npart fixed
# across this row would measure the deck, not the shader.
for n, np_ in ((64, 8192), (128, 8192), (512, 32768), (1024, 131072)):
    case("ngrid_%d" % n, edit={"field.ngrid": n, "setup.npart": np_})
case("dgrid_tight", edit={"field.dgrid": 1.0e-4})
case("dgrid_wide", edit={"field.dgrid": 6.0e-4})

# -- harmonics --------------------------------------------------------------
case("field_harm_3", edit={"track.bunchharm": 3},
     extra="&field\nharm = 3\npower = 0\ndgrid = 2.0e-4\nngrid = 256\n"
           "waist_size = 30e-6\n&end\n")
case("field_harm_1234", edit={"track.bunchharm": 4},
     extra="".join("&field\nharm = %d\npower = 0\ndgrid = 2.0e-4\nngrid = 256\n"
                   "waist_size = 30e-6\n&end\n" % h for h in (2, 3, 4)))
case("bunchharm_8", edit={"track.bunchharm": 8})

# -- the beam ---------------------------------------------------------------
case("detune_plus", edit={"setup.lambda0": 1.005e-10})
case("detune_minus", edit={"setup.lambda0": 0.995e-10})
case("spread_large", edit={"beam.delgam": 5.0})
case("current_low", edit={"beam.current": 300.0})
case("emittance_large", edit={"beam.ex": 1.2e-6, "beam.ey": 1.2e-6})
case("nbins_4", edit={"setup.nbins": 4, "setup.npart": 4096})

# -- collective effects -----------------------------------------------------
case("wake_resistive", extra="&wake\nmaterial = CU\nradius = 2.5e-3\n"
                             "roundpipe = true\ntransient = false\n&end\n")
case("wake_transient", extra="&wake\nmaterial = CU\nradius = 2.5e-3\n"
                             "roundpipe = true\ntransient = true\n&end\n")
case("wake_geometric", extra="&wake\ngap = 5\nlgap = 4.5\nradius = 2.5e-3\n&end\n")
case("wake_roughness", extra="&wake\nhrough = 1e-7\nlrough = 5e-4\n"
                             "radius = 2.5e-3\n&end\n")
case("wake_loss", extra="&wake\nloss = 12000\n&end\n")
case("spacecharge_long", extra="&efield\nlongrange = true\n&end\n")

# -- run structure ----------------------------------------------------------
case("two_track_blocks", edit={"track.zstop": 5},
     tail="&track\ngpu = true\ngpu_validate = true\nfft_fieldsolver = true\n"
          "zstop = 10\n&end\n")
case("output_step_5", edit={"track.output_step": 5})

# -- localised elements that used to fall back to the CPU --------------------
# Both run on the GPU now, so both have to stay on it for every step. They are
# kept together because they enter the step through the same branch of
# TrackBeam::track, the chicane on the opening half step and the corrector on
# the closing one.
case("chicane", lat=LAT_CHICANE)
case("corrector", lat=LAT_CORRECTOR)

# -- time dependent ---------------------------------------------------------
e, kw = td()
case("time_dependent", edit=e, **kw)
e, kw = td({"track.periodic": True})
case("time_periodic", edit=e, **kw)
e, kw = td({"setup.lambda0": 1e-10})
kw["pre"] = kw["pre"].replace("sample = 1", "sample = 4").replace(
    "slen = 2e-8", "slen = 8e-8")
case("time_sample_4", edit=e, **kw)
e, kw = td()
kw["extra"] = kw.get("extra", "") + "&wake\nmaterial = CU\nradius = 2.5e-3\n&end\n"
case("time_wake", edit=e, **kw)
e, kw = td({"track.field_dump_step": 50, "track.beam_dump_step": 50})
case("time_dumps", edit=e, **kw)

# -- physics the GPU does not implement, which must be refused --------------
case("refuse_isr_loss", extra="&sponrad\ndoLoss = true\n&end\n",
     expect="error:incoherent synchrotron radiation")
case("refuse_isr_spread", extra="&sponrad\ndoSpread = true\n&end\n",
     expect="error:incoherent synchrotron radiation")
case("refuse_spacecharge_short",
     extra="&efield\nnz = 2\nnphi = 2\nngrid = 32\n&end\n",
     expect="error:short-range space charge")
case("refuse_ngrid_odd", edit={"field.ngrid": 151},
     expect="error:Set ngrid = 128")
case("refuse_ngrid_2048", edit={"field.ngrid": 2048},
     expect="error:Set ngrid = 1024")
case("refuse_harm_5", edit={"track.bunchharm": 5},
     extra="".join("&field\nharm = %d\npower = 0\ndgrid = 2.0e-4\nngrid = 256\n"
                   "waist_size = 30e-6\n&end\n" % h for h in (2, 3, 4, 5)),
     expect="error:field harmonics")

e, kw = td({"setup.one4one": True, "setup.npart": None, "setup.nbins": None})
case("refuse_one4one", edit=e, expect="error:one4one", **kw)

# Options the GPU cannot honour. Left unguarded each of these would run to
# completion and write an output file, having quietly done something other than
# what the deck asked for, which is worse than refusing.
case("refuse_source_filter", edit={"track.source_filter": True},
     expect="error:source_filter",
     note="the two-pass field solve is exact only while the filter is off")
case("refuse_adi_solver", edit={"track.fft_fieldsolver": False},
     expect="error:fft_fieldsolver",
     note="the GPU has no ADI solver and would silently propagate by FFT")
case("refuse_bunchharm_9", edit={"track.bunchharm": 9},
     expect="error:bunchharm",
     note="beamMoments stops at 8 and the host particles are stale")

# -- the residency path, end to end -----------------------------------------
e, kw = td()
case("run_time_dependent", tier="run", edit=e, **kw)
e, kw = td({"track.field_dump_step": 50, "track.beam_dump_step": 50})
case("run_time_dumps", tier="run", edit=e, **kw)
e, kw = td()
kw["extra"] = kw.get("extra", "") + "&wake\nmaterial = CU\nradius = 2.5e-3\n&end\n"
case("run_time_wake", tier="run", edit=e, **kw)
case("run_two_track_blocks", tier="run", edit={"track.zstop": 5},
     tail="&track\ngpu = true\nfft_fieldsolver = true\nzstop = 10\n&end\n")
case("run_planar_saturation", tier="run", lat=lattice(hel=False),
     edit={"track.zstop": 40})
case("run_ngrid_1024", tier="run",
     edit={"field.ngrid": 1024, "setup.npart": 131072})
e, kw = td()
case("run_time_saturation", tier="run", edit=dict(e, **{"track.zstop": 40}), **kw)
# The chicane writes the resident particle phases through the R56 shear, and it
# does so once rather than on every step, which is the pattern the step tier is
# least able to police: it resynchronises either side of every step, so a phase
# written to the wrong copy would be repaired before it could be seen.
case("run_chicane", tier="run", lat=LAT_CHICANE)
case("run_corrector", tier="run", lat=LAT_CORRECTOR)


# --------------------------------------------------------------------------


def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def render(case, name, gpu):
    """Build the deck text. gpu is True, False or "validate"."""
    groups = [(n, dict(d)) for n, d in BASE]

    for key, val in case.edit.items():
        ns, k = key.split(".", 1)
        for gname, gdict in groups:
            if gname == ns:
                if val is None:
                    gdict.pop(k, None)
                else:
                    gdict[k] = val
                break
        else:
            raise KeyError("no &%s in the base deck (%s)" % (ns, key))

    for gname, gdict in groups:
        if gname == "setup":
            gdict["rootname"] = name
            gdict["lattice"] = case.name + ".lat"
        if gname == "track":
            gdict["gpu"] = gpu is not False
            gdict["gpu_validate"] = gpu == "validate"

    out = []
    for gname, gdict in groups:
        if gname == "field":
            out.append(case.pre)
        if gname == "track":
            out.append(case.extra)
        out.append("&%s\n" % gname)
        out.extend("%s = %s\n" % (k, fmt(v)) for k, v in gdict.items())
        out.append("&end\n\n")

    tail = case.tail
    if tail and gpu is not True:
        tail = tail.replace("gpu_validate = true\n", "")
        if gpu is False:
            tail = tail.replace("gpu = true", "gpu = false")
    out.append(tail)

    text = "".join(out)
    if gpu is False:
        text = text.replace("gpu = false\n", "")
    return text


# The backend names itself in both lines, so these match whichever one the
# binary was built with rather than Metal specifically.
ERR_LINE = re.compile(
    r"\w+ vs CPU over (\d+) steps: max relative error, field (\S+), beam (\S+)")
FALLBACK = re.compile(r"\w+: (\d+) of (\d+) steps fell back")


def run(genesis, mpirun, workdir, case, name, gpu, ranks):
    deck = os.path.join(workdir, name + ".in")
    with open(deck, "w") as f:
        f.write(render(case, name, gpu))
    cmd = ([mpirun, "-n", str(ranks)] if ranks > 1 else []) + [genesis, name + ".in"]
    env = dict(os.environ, FI_PROVIDER="tcp")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=workdir, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    return p.stdout, time.time() - t0


def compare(workdir, a, b):
    p = subprocess.run([sys.executable, os.path.join(HERE, "compare.py"),
                        a + ".out.h5", b + ".out.h5"],
                       cwd=workdir, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0 and "worst of" not in p.stdout:
        return None, p.stdout
    m = re.search(r"worst of \d+: (\S+)", p.stdout)
    return (float(m.group(1)) if m else None), p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genesis", default=os.path.join(HERE, "..", "..",
                                                      "build-metal", "genesis4"))
    ap.add_argument("--mpirun", default=shutil.which("mpirun") or "mpirun")
    ap.add_argument("--workdir", default="/tmp/g4sweep")
    ap.add_argument("--only", default="", help="run only cases matching this regex")
    ap.add_argument("--tier", default="", choices=["", "step", "run"])
    ap.add_argument("--field-tol", type=float, default=5e-3)
    ap.add_argument("--beam-tol", type=float, default=5e-4)
    ap.add_argument("--run-margin", type=float, default=2.0,
                    help="how many times the accumulated per-step round-off "
                         "the end-to-end difference may reach")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    genesis = os.path.abspath(args.genesis)
    if not os.path.exists(genesis):
        sys.exit("no genesis4 at " + genesis)
    os.makedirs(args.workdir, exist_ok=True)

    sel = re.compile(args.only) if args.only else None
    cases = [c for c in CASES
             if (sel is None or sel.search(c.name))
             and (not args.tier or c.tier == args.tier)]

    # The two tiers report different quantities in the two numeric columns, so
    # each gets its own header rather than one heading that is wrong for half
    # the table.
    HEADINGS = {"step": ("field", "beam"), "run": ("vs CPU", "bound")}
    shown = None

    bad = []
    for c in cases:
        if c.tier != shown:
            shown = c.tier
            print("%-26s %-5s %10s %10s  %s"
                  % (("case", "tier") + HEADINGS[shown] + ("verdict",)))
            print("-" * 78)
        with open(os.path.join(args.workdir, c.name + ".lat"), "w") as f:
            f.write(c.lat)

        want_error = c.expect.startswith("error:")
        verdict, field, beam = "", None, None

        if c.tier == "step" or want_error:
            out, secs = run(genesis, args.mpirun, args.workdir, c,
                            c.name, "validate", c.ranks)
            if want_error:
                needle = c.expect.split(":", 1)[1]
                if needle.lower() in out.lower():
                    verdict = "refused, as it should be"
                else:
                    verdict = "NOT REFUSED (wanted %r)" % needle
            elif "*** Error" in out:
                verdict = "ERROR: " + next(
                    (l.strip() for l in out.splitlines() if "*** Error" in l), "?")
            else:
                m = ERR_LINE.search(out)
                if not m:
                    verdict = "no comparison line in the output"
                else:
                    field, beam = float(m.group(2)), float(m.group(3))
                    fb = FALLBACK.search(out)
                    if c.expect == "fallback" and not fb:
                        verdict = "expected a CPU fallback and got none"
                    elif fb and c.expect != "fallback":
                        # Silently dropping to the CPU is a correct answer
                        # obtained the slow way, and it hides an unported
                        # element, so it counts as a failure here.
                        verdict = "UNEXPECTED CPU FALLBACK, %s/%s steps" % (
                            fb.group(1), fb.group(2))
                    elif field > args.field_tol or beam > args.beam_tol:
                        verdict = "OVER TOLERANCE"
                    else:
                        verdict = "ok"
                        if fb:
                            verdict += ", %s/%s steps on the CPU" % (
                                fb.group(1), fb.group(2))
        else:
            cpu = c.name + "_cpu"
            gpu = c.name + "_gpu"
            chk = c.name + "_chk"
            out_c, _ = run(genesis, args.mpirun, args.workdir, c, cpu, False, c.ranks)
            out_g, _ = run(genesis, args.mpirun, args.workdir, c, gpu, True, c.ranks)
            # The bound: the same deck under gpu_validate, which resynchronises
            # every step and so reports the round-off of one step in isolation.
            # Times the step count, that is the most that round-off alone can
            # produce end to end.
            out_v, _ = run(genesis, args.mpirun, args.workdir, c, chk,
                           "validate", c.ranks)
            m = ERR_LINE.search(out_v)
            bound = float(m.group(2)) * int(m.group(1)) if m else None
            if "*** Error" in out_g:
                verdict = "ERROR: " + next(
                    (l.strip() for l in out_g.splitlines() if "*** Error" in l), "?")
            elif "*** Error" in out_c:
                verdict = "the CPU reference did not run"
            elif bound is None:
                verdict = "gpu_validate reported no per-step error"
            else:
                worst, text = compare(args.workdir, cpu, gpu)
                if worst is None:
                    verdict = "comparison failed"
                    print(text)
                else:
                    field, beam = worst, bound
                    verdict = ("ok" if worst <= args.run_margin * bound
                               else "OVER TOLERANCE")

        print("%-26s %-5s %10s %10s  %s" % (
            c.name, c.tier,
            "" if field is None else "%.3e" % field,
            "" if beam is None else "%.3e" % beam,
            verdict))
        if verdict != "ok" and not verdict.startswith(("ok,", "refused")):
            bad.append((c, verdict))
            if not args.keep_going:
                pass

    print("-" * 78)
    if not bad:
        print("all %d cases passed" % len(cases))
        return 0
    print("%d of %d cases need attention:" % (len(bad), len(cases)))
    for c, v in bad:
        print("  %-26s %s%s" % (c.name, v, "   (" + c.note + ")" if c.note else ""))
    return 1


if __name__ == "__main__":
    sys.exit(main())
