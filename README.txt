Membrane pre-concentration ahead of MEA absorption: data and code
=================================================================

Contents
--------
MEA_Validation_Kaiserslautern.apwz
    Aspen Plus V14. Rate-based MEA model (ENRTL-RK, RateSep) validated against
    case 1 of the Kaiserslautern pilot plant (Notz et al., 2012). Closed loop with
    water and MEA make-up balances. Results: Table 3 of the paper.

MEA_NGCC_Reference.apwz
    Aspen Plus V14. Industrial MEA reference plant for one of four trains of NETL
    Case B31B.90 (NGCC, 90 % capture): blower, DCC, absorber (11.3 m, 20 m packing),
    stripper (4.6 m, 15 m packing, 2.0 bar), lean/rich exchanger (dTmin = 10 K),
    treated-gas condenser, closed solvent loop, CO2 compression to 15.3 MPa.
    Results: Section 3.2 of the paper.
    Note: this file also contains a DEACTIVATED copy of the pilot validation
    flowsheet (identical to MEA_Validation_Kaiserslautern.apwz); only the NGCC
    blocks are active.

    Amine performance map (Table 4): map points were generated from a copy of this
    file run open-loop (lean solvent as a fixed feed at loading 0.20, design specs on
    90 % capture and on CO2 release in the stripper), changing only the flue gas
    N2/O2/Ar flows (CO2 flow constant), the absorber diameter (about 70 % flooding)
    and the exchanger cold-end approach (internal approach >= 10 K), as listed in
    Table 4. The map values are also provided in results/amine_map_aspen.csv,
    written by the Python script.

hybrid_membrane_mea_analysis.py
    Single Python script reproducing all Python-based results and Figs 2-6:
      Stage 0  reproduction of the module-level results of Ajoku et al. (preprint)
      Stage 1  hybrid design: membrane sized for 94.7 % CO2 recovery, pure-gas vs
               mixed-gas design, NGCC / coal / cement, four pressure configurations,
               F.K = 0.123 / 0.191 / 0.272
      Stage 2  cost of compared items (base and optimistic assumptions)
      Stage 3  sensitivity (one-at-a-time and all-favourable cases)
      Stage 4  CO2 compression cross-check with the Span-Wagner EOS (CoolProp)
      Stage 5  Figures 2-6 (600 dpi PNG and vector PDF)

How to run
----------
    pip install numpy pandas scipy matplotlib CoolProp
    python hybrid_membrane_mea_analysis.py             # full analysis (about 20-40 min)
    python hybrid_membrane_mea_analysis.py --quick     # smoke test (about 1 min)
    python hybrid_membrane_mea_analysis.py --from-csv  # reuse Stage 1 results, redo the rest

Outputs: results/*.csv and figures/*.png|pdf (created automatically).
Fig. 1 (process schematic) was drawn separately and is not produced by the script.

Software
--------
Aspen Plus V14 (AspenTech; licence required to open .apwz compound files; saved in V14 and may not open in earlier versions).
Python 3.9 or later with numpy, pandas, scipy, matplotlib, CoolProp.

Not included
------------
AspenTech's ENRTL-RK_Rate_Based_MEA_Model example file and documentation (copyright
AspenTech; cited in the paper) and the NETL baseline report (publicly available).

Reference for the membrane model
--------------------------------
M. V. Ajoku, M. Sola-Ojo, and G. C. Duru, "Can pure-gas permeability data design a
mixed-gas module? A correctability map for dual-mode CO2/N2 transport in Matrimid
5218 hollow fibres," preprint, 2026, doi: [to be added].
