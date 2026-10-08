#!/usr/bin/env python3
# =============================================================================
#  Membrane pre-concentration ahead of MEA absorption:
#  hybrid membrane-amine analysis for NGCC, coal and cement flue gases
#
#  Single script reproducing every Python-based result in the paper:
#    Stage 0  Model check   : reproduces the module-level results of [Ref. 8]
#    Stage 1  Hybrid design : membrane sized for 94.7 % CO2 recovery, pure-gas (A)
#                             vs mixed-gas (B) design, three sources, four pressure
#                             configurations, F.K = 0.123 / 0.191 / 0.272   (Tables 5, Figs 3-4)
#    Stage 2  Cost          : annualised cost of compared items, base and optimistic   (Fig. 5)
#    Stage 3  Sensitivity   : one-at-a-time and all-favourable cases                   (Fig. 6)
#    Stage 4  Compression   : CO2 compression cross-check with Span-Wagner (CoolProp)
#    Stage 5  Figures       : Figs 2-6, 600 dpi PNG + vector PDF
#
#  Amine-plant inputs (SRD, absorber diameter, rich loading/temperature) come from the
#  Aspen Plus V14 rate-based MEA model (MEA_NGCC_Reference.bkp; map points in Table 4).
#  Membrane model: competitive dual-mode, counter-current hollow fibre, Matrimid 5218,
#  as developed and calibrated in [Ref. 8] (Ajoku et al., preprint, 2026).
#
#  Usage:   python hybrid_membrane_mea_analysis.py            (all stages)
#           python hybrid_membrane_mea_analysis.py --quick    (smoke test, ~1 min)
#  Outputs: ./results/*.csv  and  ./figures/*.png|pdf
#  Requirements: Python >= 3.9, numpy, pandas, scipy, matplotlib; CoolProp (Stage 4 only)
#  Run time: Stage 1 dominates (about 20-40 min on a single CPU core); others take seconds.
# =============================================================================
import os, sys, copy, argparse
import numpy as np, pandas as pd
from scipy.optimize import brentq

os.makedirs('results', exist_ok=True)

# =============================================================================
# 1. MEMBRANE MODEL (unchanged from Ref. 8)
# =============================================================================
BAR_TO_CMHG = 75.0064
ATM = 1.01325
MOLAR_VOL = 22414.0

def barrer_to_SI(P_barrer, L_cm):
    """Permeability (Barrer) and thickness (cm) -> permeance (mol m-2 s-1 bar-1)."""
    return P_barrer * 1e-10 * BAR_TO_CMHG / L_cm / MOLAR_VOL * 1e4

def make_module(n_fibers, l_membrane_m, L=1.0, d_outer=3e-4):
    a = n_fibers * np.pi * d_outer
    return {'L': L, 'n_fibers': n_fibers, 'a': a, 'A_total': a * L,
            'L_cm': l_membrane_m * 100.0, 'l_membrane_m': l_membrane_m}

def make_matrimid_T1(F_N2=0.0, P_N2=0.19, bN2=None, FK_CO2=None):
    """Dual-mode parameters (Moore & Koros 2007) with D_D, D_H set by F.K (Ref. 8)."""
    kD = {'CO2': 1.42, 'N2': 0.120 / ATM}
    CH = {'CO2': 25.5, 'N2': 3.94}
    b = {'CO2': 0.367, 'N2': 0.087 if bN2 is None else bN2}
    DD_c, DH_c = 2.141e-8, 2.79e-9
    K_c = CH['CO2'] * b['CO2'] / kD['CO2']
    if FK_CO2 is not None:                      # keep P(CO2, 4 bar) = 5.464 Barrer
        p_keep = 4.0
        P_ref = (kD['CO2'] / BAR_TO_CMHG) * DD_c * 1e10 * (1 + (DH_c / DD_c) * K_c / (1 + b['CO2'] * p_keep))
        F_c = FK_CO2 / K_c
        DD_c = P_ref / ((kD['CO2'] / BAR_TO_CMHG) * 1e10 * (1 + F_c * K_c / (1 + b['CO2'] * p_keep)))
        DH_c = DD_c * F_c
    KN = CH['N2'] * b['N2'] / kD['N2']
    DD_n = P_N2 / ((kD['N2'] / BAR_TO_CMHG) * 1e10 * (1.0 + F_N2 * KN))
    return {'kD': kD, 'CH': CH, 'b': b,
            'DD': {'CO2': DD_c, 'N2': DD_n}, 'DH': {'CO2': DH_c, 'N2': DD_n * F_N2},
            'F': {'CO2': DH_c / DD_c, 'N2': F_N2}, 'P_pure': {'CO2': 5.5, 'N2': P_N2}}

def _sorb(m, gas, p_i, p_j, other):
    kD, CH = m['kD'][gas], m['CH'][gas]
    bi, bj = m['b'][gas], m['b'][other]
    return kD * p_i, CH * bi * p_i / (1.0 + bi * p_i + bj * p_j)        # Eq. (2)

def P_eff_exact(m, gas, pf_i, pf_j, pp_i, pp_j, other):
    """Mixed-gas permeability (Barrer), Eq. (3)."""
    dp = pf_i - pp_i
    if dp <= 0.0:
        return 0.0
    CDf, CHf = _sorb(m, gas, pf_i, pf_j, other)
    CDp, CHp = _sorb(m, gas, pp_i, pp_j, other)
    Nl = m['DD'][gas] * (CDf - CDp) + m['DH'][gas] * (CHf - CHp)
    return 1e10 * Nl / (dp * BAR_TO_CMHG)

def P_vac(m, gas, p, p_other=0.0):
    other = 'N2' if gas == 'CO2' else 'CO2'
    return P_eff_exact(m, gas, p, p_other, 0.0, 0.0, other)

def get_J(m, model, yc, xc, p_feed, p_perm, SI):
    """Local fluxes. model 'A' = pure-gas permeabilities, 'B' = mixed-gas."""
    yc = float(np.clip(yc, 1e-9, 1 - 1e-9)); xc = float(np.clip(xc, 1e-9, 1 - 1e-9))
    pcf, pnf = yc * p_feed, (1 - yc) * p_feed
    pcp, pnp = xc * p_perm, (1 - xc) * p_perm
    if model == 'A':
        Pc, Pn = m['P_pure']['CO2'], m['P_pure']['N2']
    else:
        Pc = P_eff_exact(m, 'CO2', pcf, pnf, pcp, pnp, 'N2')
        Pn = P_eff_exact(m, 'N2', pnf, pcf, pnp, pcp, 'CO2')
    return SI * Pc * max(pcf - pcp, 0.0), SI * Pn * max(pnf - pnp, 0.0)

def run_module(m, model, cond, module, N=50, max_iter=20000, tol=1e-9, omega=0.4):
    """Counter-current hollow-fibre module, Eq. (4), under-relaxed successive substitution."""
    F_in, y_in = cond['F_feed_in'], cond['y_CO2_feed']
    p_feed, p_perm = cond['p_feed'], cond['p_perm']
    L, a = module['L'], module['a']
    SI = barrer_to_SI(1.0, module['L_cm']); dz = L / N
    Fc_in, Fn_in = F_in * y_in, F_in * (1.0 - y_in)
    floor = F_in * 1e-9
    Pc0, Pn0 = m['P_pure']['CO2'], m['P_pure']['N2']
    Jc0 = SI * Pc0 * (y_in * (p_feed - p_perm)); Jn0 = SI * Pn0 * ((1 - y_in) * (p_feed - p_perm))
    xc_est = Jc0 / max(Jc0 + Jn0, 1e-30)
    Fp_est = min((Jc0 + Jn0) * a * L * 0.5, F_in * 0.4)
    Fcf = np.linspace(Fc_in, max(Fc_in - Fp_est * xc_est, Fc_in * 0.5), N)
    Fnf = np.linspace(Fn_in, max(Fn_in - Fp_est * (1 - xc_est), Fn_in * 0.5), N)
    Fcp = np.linspace(Fp_est * xc_est, 0.0, N)
    Fnp = np.linspace(Fp_est * (1 - xc_est), 0.0, N)
    for _ in range(max_iter):
        o = [Fcf.copy(), Fnf.copy(), Fcp.copy(), Fnp.copy()]
        fc, fn = Fc_in, Fn_in
        for k in range(N):
            yc = Fcf[k] / max(Fcf[k] + Fnf[k], 1e-20); xc = Fcp[k] / max(Fcp[k] + Fnp[k], 1e-20)
            Jc, Jn = get_J(m, model, yc, xc, p_feed, p_perm, SI)
            Fcf[k] = omega * max(fc - dz * a * Jc, floor) + (1 - omega) * o[0][k]
            Fnf[k] = omega * max(fn - dz * a * Jn, floor) + (1 - omega) * o[1][k]
            fc, fn = Fcf[k], Fnf[k]
        pc, pn = 0.0, 0.0
        for k in range(N - 1, -1, -1):
            yc = Fcf[k] / max(Fcf[k] + Fnf[k], 1e-20); xc = Fcp[k] / max(Fcp[k] + Fnp[k], 1e-20)
            Jc, Jn = get_J(m, model, yc, xc, p_feed, p_perm, SI)
            Fcp[k] = omega * max(pc + dz * a * Jc, 0.0) + (1 - omega) * o[2][k]
            Fnp[k] = omega * max(pn + dz * a * Jn, 0.0) + (1 - omega) * o[3][k]
            pc, pn = Fcp[k], Fnp[k]
        if sum(np.max(np.abs(A - B)) for A, B in zip([Fcf, Fnf, Fcp, Fnp], o)) < tol:
            break
    Fcp_out, Fnp_out = float(Fcp[0]), float(Fnp[0])
    return {'CO2_recovery': Fcp_out / Fc_in * 100,
            'CO2_purity': Fcp_out / max(Fcp_out + Fnp_out, 1e-20) * 100,
            'stage_cut': (Fcp_out + Fnp_out) / F_in * 100,
            'mb_error_pct': abs(Fc_in - Fcp_out - float(Fcf[-1])) / Fc_in * 100}

def run_module_robust(m, model, cond, module):
    """Tolerance relative to the module feed flow (needed for low-pressure industrial cases)."""
    return run_module(m, model, cond, module, tol=1e-8 * cond['F_feed_in'])

P_REF_CO2, P_REF_N2 = 5.5, 0.19
def design_basis(m, p_ref):
    """Pure-gas design data: CO2 permeability measured at p_ref (bar)."""
    d = copy.deepcopy(m); d['P_pure']['CO2'] = P_vac(m, 'CO2', p_ref); return d

# =============================================================================
# 2. PLANT BASIS, AMINE MAP (Aspen Plus) AND ASSUMPTIONS
# =============================================================================
CO2_KMOLH = 1411.74                               # CO2 per train (NETL B31B.90 / 4)
FEED_Y = {'NGCC': 0.0447, 'Coal': 0.1466, 'Cement': 0.200}   # dry CO2 fraction [9], [10]
PRESSURES = [(1.1, 0.11), (2.0, 0.20), (5.0, 1.0), (10.0, 1.0)]
FKS = [0.123, 0.191, 0.272]                       # F.K interval from Ref. 8
CAP_OVERALL, CAP_AMINE = 0.90, 0.95
R_MEM = CAP_OVERALL / CAP_AMINE                   # 0.947
T_CO2 = CAP_OVERALL * CO2_KMOLH * 44.01 / 1000    # t/h captured
F_NGCC = (1411.74 + 30162.12) / 3.6               # NGCC dry flue gas per train, mol/s
W_BLOWER_NGCC = 1.72                              # MW, Aspen reference plant

MOD_IND = make_module(10_000, 0.1e-6)             # industrial skin 0.1 um (area only)
SI_IND = barrer_to_SI(1.0, MOD_IND['L_cm'])

# Amine performance map (Aspen Plus, Table 4); SRD and D held constant above 40 %
MAP = pd.DataFrame(dict(x=[4.47, 6, 10, 15, 20, 30, 40],
                        SRD=[3.88, 3.83, 3.75, 3.69, 3.64, 3.59, 3.58],
                        D=[11.3, 10.3, 8.5, 7.5, 7.0, 6.5, 6.0],
                        rich=[0.481, 0.485, 0.493, 0.500, 0.506, 0.514, 0.516],
                        Trich=[41.3, 41.4, 42.2, 44.6, 47.4, 52.4, 56.6]))
MAP.to_csv('results/amine_map_aspen.csv', index=False)

R_GAS, T_GAS, K_GAS = 8.314, 313.15, 1.4
HOURS, D_REF, ABS_EXP = 8000, 11.3, 1.4
BASE = dict(th=0.25, elec=60, rot=1000, abs=22e6, mem=50, life=5, crf=0.0937,
            eta_c=0.80, eta_v=0.70, eta_e=0.85, srd_hyb=1.0)
OPTIMISTIC = {**BASE, 'mem': 20, 'life': 8, 'rot': 600}
RANGES = dict(th=[0.15, 0.35], elec=[30, 120], rot=[500, 1500], abs=[11e6, 66e6],
              mem=[0, 100], life=[3, 10], crf=[0.06, 0.12],
              eta_c=[0.70, 0.88], eta_v=[0.60, 0.80], eta_e=[0.75, 0.90], srd_hyb=[0.95, 1.05])
FAVOURABLE = dict(th=0.35, elec=120, rot=500, abs=66e6, mem=0, life=10, crf=0.06,
                  eta_c=0.88, eta_v=0.80, eta_e=0.90, srd_hyb=0.95)
CARBON_PRICE = 100.0                              # USD/t, value of capture shortfall

# =============================================================================
# 3. MACHINES, ENERGY AND COST (Eqs. 6-9)
# =============================================================================
def comp_MW(n, p1, p2, eta, rmax=3.0):
    """Intercooled ideal-gas compressor / vacuum pump, n in mol/s, Eq. (6)."""
    if p2 <= p1 * 1.001: return 0.0
    N = int(np.ceil(np.log(p2 / p1) / np.log(rmax))); r = (p2 / p1) ** (1 / N)
    return N * n * R_GAS * T_GAS * K_GAS / (K_GAS - 1) * (r ** ((K_GAS - 1) / K_GAS) - 1) / eta / 1e6

def exp_MW(n, p1, p2, eta):
    if p1 <= p2 * 1.001: return 0.0
    return n * R_GAS * T_GAS * K_GAS / (K_GAS - 1) * (1 - (p2 / p1) ** ((K_GAS - 1) / K_GAS)) * eta / 1e6

COMP = ['Steam (power equivalent)', 'Machinery power', 'Machinery capital',
        'Membrane capital + replacement', 'Absorber capital']

def breakdown(feed, row, p, hybrid=True):
    """Annualised cost of compared items, USD per t CO2 (Eqs. 8-9)."""
    y = FEED_Y[feed]; F_TR = CO2_KMOLH / y / 3.6; ann = p['crf'] + 0.04; t = T_CO2 * HOURS
    if hybrid:
        n_perm = R_MEM * CO2_KMOLH / 3.6 / (row.PurB / 100)
        Wc = comp_MW(F_TR, 1.0, row.pf, p['eta_c']); Wv = comp_MW(n_perm, row.pp, 1.05, p['eta_v'])
        We = exp_MW(F_TR - n_perm, row.pf, 1.013, p['eta_e'])
        Wnet, Wcap = Wc + Wv - We, Wc + Wv + We
        srd = np.interp(row.PurB, MAP.x, MAP.SRD) * p['srd_hyb']
        D = np.interp(row.PurB, MAP.x, MAP.D); area = row.AreaB_Mm2 * 1e6
    else:
        Wnet = Wcap = W_BLOWER_NGCC * F_TR / F_NGCC
        srd = np.interp(y * 100, MAP.x, MAP.SRD); D = np.interp(y * 100, MAP.x, MAP.D); area = 0.0
    return dict(zip(COMP, [p['th'] * srd * T_CO2 / 3.6 * HOURS * p['elec'] / t, Wnet * HOURS * p['elec'] / t,
                           ann * Wcap * 1e3 * p['rot'] / t, (ann * area * p['mem'] + area * p['mem'] / p['life']) / t,
                           ann * p['abs'] * (D / D_REF) ** ABS_EXP / t]))

def dcost(feed, row, p):
    return sum(breakdown(feed, row, p).values()) - sum(breakdown(feed, None, p, False).values())

# =============================================================================
# STAGE 0: reproduce module-level results of Ref. 8 (10/1 bar, design data at 1 bar)
# =============================================================================
def stage0():
    MOD = make_module(10_000, 50e-6); SI = barrer_to_SI(1.0, MOD['L_cm'])
    m = make_matrimid_T1(FK_CO2=0.191); mA = design_basis(m, 1.0); rows = []
    for y in [0.10, 0.15]:
        Fin = max(SI * (P_REF_CO2 * y + P_REF_N2 * (1 - y)) * (10.0 - 1.0) * MOD['a'] * MOD['L'] * 5.0, 1e-5)   # PHI_BASE = 5 in Ref. 8
        c = {'F_feed_in': Fin, 'y_CO2_feed': y, 'p_feed': 10.0, 'p_perm': 1.0}
        rA = run_module(mA, 'A', c, MOD); rB = run_module(m, 'B', c, MOD)
        rows.append(dict(y_feed=y, RecA=rA['CO2_recovery'], RecB=rB['CO2_recovery'],
                         dRec=rA['CO2_recovery'] - rB['CO2_recovery'], dPur=rA['CO2_purity'] - rB['CO2_purity']))
    df = pd.DataFrame(rows); df.to_csv('results/stage0_reproduction.csv', index=False)
    print('\nStage 0  reproduction of Ref. 8 (expected dRec ~2.5 points, dPur ~1.2 points)')
    print(df.round(3).to_string(index=False))

# =============================================================================
# STAGE 1: hybrid design across sources (pure-gas A vs mixed-gas B)
# =============================================================================
def F_in_ind(y, pf, pp, phi):
    return SI_IND * (P_REF_CO2 * y + P_REF_N2 * (1 - y)) * (pf - pp) * MOD_IND['a'] * MOD_IND['L'] * phi  # Eq. (5)

def run_ind(mm, model, y, pf, pp, phi):
    c = {'F_feed_in': F_in_ind(y, pf, pp, phi), 'y_CO2_feed': y, 'p_feed': pf, 'p_perm': pp}
    return run_module_robust(mm, model, c, MOD_IND)

def phi_for_R(mm, model, y, pf, pp):
    f = lambda lp: run_ind(mm, model, y, pf, pp, np.exp(lp))['CO2_recovery'] - R_MEM * 100
    return float(np.exp(brentq(f, np.log(0.2), np.log(8.0), xtol=2e-3)))

def stage1(feeds, pressures, fks):
    mA = design_basis(make_matrimid_T1(FK_CO2=0.191), 1.0); rows = []
    for feed in feeds:
        y = FEED_Y[feed]; F_TR = CO2_KMOLH / y / 3.6
        srd_mea = float(np.interp(y * 100, MAP.x, MAP.SRD))
        E_MEA = W_BLOWER_NGCC * F_TR / F_NGCC + BASE['th'] * srd_mea * T_CO2 / 3.6
        for pf, pp in pressures:
            phiA = phi_for_R(mA, 'A', y, pf, pp)
            for fk in fks:
                mB = make_matrimid_T1(FK_CO2=fk)
                phiB = phi_for_R(mB, 'B', y, pf, pp)
                rB, rAB = run_ind(mB, 'B', y, pf, pp, phiB), run_ind(mB, 'B', y, pf, pp, phiA)
                aA = F_TR / F_in_ind(y, pf, pp, phiA) * MOD_IND['A_total'] / 1e6
                aB = F_TR / F_in_ind(y, pf, pp, phiB) * MOD_IND['A_total'] / 1e6
                n_perm = rB['stage_cut'] / 100 * F_TR
                Wc = comp_MW(F_TR, 1.0, pf, BASE['eta_c'])
                Wv = comp_MW(n_perm, pp, 1.05, BASE['eta_v'])
                We = exp_MW(F_TR - n_perm, pf, 1.013, BASE['eta_e'])
                srd = float(np.interp(rB['CO2_purity'], MAP.x, MAP.SRD))
                E = Wc + Wv - We + BASE['th'] * srd * T_CO2 / 3.6
                rows.append(dict(feed=feed, pf=pf, pp=pp, FK=fk, PurB=rB['CO2_purity'],
                                 StageCutB=rB['stage_cut'], AreaB_Mm2=aB, Under_pct=(aA / aB - 1) * 100,
                                 CapIfA=rAB['CO2_recovery'] * CAP_AMINE, SRD=srd, Wcomp=Wc, Wvac=Wv, Wexp=We,
                                 E_MWe=E, E_MEA=E_MEA, dE_pct=(E / E_MEA - 1) * 100,
                                 MB_final=max(rB['mb_error_pct'], rAB['mb_error_pct'])))
            print(f'  done {feed} {pf}/{pp} bar')
    df = pd.DataFrame(rows); df.to_csv('results/stage1_hybrid_design.csv', index=False)
    print('\nStage 1  hybrid design (FK = 0.191 rows shown; full table in results/)')
    print(df[np.isclose(df.FK, 0.191)].round(2).to_string(index=False))
    return df

# =============================================================================
# STAGE 2: cost comparison (base and optimistic)
# =============================================================================
def stage2(df):
    B = df[np.isclose(df.FK, 0.191)]; rows = []
    for sname, p in [('base', BASE), ('optimistic', OPTIMISTIC)]:
        for r in B.itertuples():
            hyb, mea = breakdown(r.feed, r, p), breakdown(r.feed, None, p, False)
            g = df[(df.feed == r.feed) & (df.pf == r.pf)]
            miss = (CAP_OVERALL * 100 - g.CapIfA.min()) / 100 * CO2_KMOLH * 44.01 / 1000 * HOURS
            rows.append(dict(scenario=sname, feed=r.feed, pf=r.pf, pp=r.pp,
                             cost_hybrid=sum(hyb.values()), cost_MEA=sum(mea.values()),
                             d_cost_per_t=sum(hyb.values()) - sum(mea.values()),
                             PG_shortfall_MUSD_y=max(miss, 0) * CARBON_PRICE / 1e6,
                             **{('hyb_' + k): v for k, v in hyb.items()}))
    out = pd.DataFrame(rows); out.to_csv('results/stage2_costs.csv', index=False)
    print('\nStage 2  cost of compared items (USD/t CO2)')
    print(out[['scenario', 'feed', 'pf', 'pp', 'cost_hybrid', 'cost_MEA', 'd_cost_per_t', 'PG_shortfall_MUSD_y']]
          .round(2).to_string(index=False))

# =============================================================================
# STAGE 3: sensitivity
# =============================================================================
def stage3(df):
    B = df[np.isclose(df.FK, 0.191)]; rows = []
    for feed in B.feed.unique():
        sub = B[B.feed == feed]
        best = min(sub.itertuples(), key=lambda r: dcost(feed, r, BASE))
        for k, (lo, hi) in RANGES.items():
            rows.append(dict(feed=feed, pf=best.pf, parameter=k, base=dcost(feed, best, BASE),
                             at_low=dcost(feed, best, {**BASE, k: lo}), at_high=dcost(feed, best, {**BASE, k: hi})))
        fav = [(r.pf, r.pp, dcost(feed, r, FAVOURABLE)) for r in sub.itertuples()]
        b = min(fav, key=lambda v: v[2])
        print(f'  {feed}: all parameters favourable to hybrid -> {b[2]:.1f} USD/t at {b[0]}/{b[1]} bar')
    out = pd.DataFrame(rows); out.to_csv('results/stage3_sensitivity.csv', index=False)
    print('\nStage 3  one-at-a-time sensitivity (USD/t CO2)')
    print(out.round(1).to_string(index=False))

# =============================================================================
# STAGE 4: CO2 compression cross-check (Span-Wagner via CoolProp)
# =============================================================================
def stage4():
    try:
        import CoolProp.CoolProp as CP
    except ImportError:
        print('\nStage 4 skipped (pip install CoolProp)'); return
    m = 1270.44 / 3.6 * 44.0095 / 1000                             # kg/s (stripper CO2 product)
    P, T, W = 2.0e5, 313.15, 0.0
    r = (153e5 / 2.0e5) ** (1 / 5)
    for i in range(5):
        h1 = CP.PropsSI('H', 'P', P, 'T', T, 'CO2'); s1 = CP.PropsSI('S', 'P', P, 'T', T, 'CO2')
        h2s = CP.PropsSI('H', 'P', P * r, 'S', s1, 'CO2')
        W += m * (h2s - h1) / 0.80 / 1e3; P *= r; T = 313.15
    rho = CP.PropsSI('D', 'P', 153e5, 'T', 303.15, 'CO2')
    print(f'\nStage 4  CO2 compression 2 -> 153 bar, 5 stages, eta 0.80: {W:.0f} kW '
          f'(Aspen 4968 kW; NETL 4475 kW per train); outlet density {rho:.0f} kg/m3')

# =============================================================================
# STAGE 5: figures 2-6
# =============================================================================
def stage5(df):
    import matplotlib as mpl, matplotlib.pyplot as plt, shutil
    os.makedirs('figures', exist_ok=True)
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'axes.linewidth': 0.7, 'lines.linewidth': 1.3, 'lines.markersize': 4,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'pdf.fonttype': 42})
    C = {'NGCC': '#0072B2', 'Coal': '#D55E00', 'Cement': '#009E73'}
    MK = {'NGCC': 'o', 'Coal': 's', 'Cement': '^'}
    UNIT = 'USD t$^{-1}$ CO$_2$'
    CFG_LAB = ['1.1 / 0.11', '2 / 0.2', '5 / 1', '10 / 1']; xs = np.arange(len(PRESSURES))
    B = df[np.isclose(df.FK, 0.191)]
    best = {f: min(B[B.feed == f].itertuples(), key=lambda r: dcost(f, r, BASE)) for f in FEED_Y}

    def save(fig, name):
        fig.savefig(f'figures/{name}.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
        fig.savefig(f'figures/{name}.pdf', bbox_inches='tight', pad_inches=0.03); plt.close(fig)
    def letter(ax, s, x=-0.16, y=1.04):
        ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')

    # Fig. 2  amine map
    with mpl.rc_context({'font.size': 6, 'axes.labelsize': 6, 'xtick.labelsize': 6,
                         'ytick.labelsize': 6, 'legend.fontsize': 6}):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
        a1.axvspan(20, 40, color='0.92', zorder=0)
        l1, = a1.plot(MAP.x, MAP.SRD, 'o-', c='k'); a1.set_ylim(3.50, 3.95)
        a1.set_xlabel('CO$_2$ at amine inlet (%, dry)'); a1.set_ylabel('Specific reboiler duty (GJ t$^{-1}$)')
        t1 = a1.twinx(); t1.spines['right'].set_visible(True)
        l2, = t1.plot(MAP.x, MAP.D, 's--', c='#0072B2'); t1.set_ylim(4.5, 12.5)
        t1.set_ylabel('Absorber diameter (m)', color='#0072B2'); t1.tick_params(axis='y', colors='#0072B2')
        a1.legend([l1, l2], ['Specific reboiler duty', 'Absorber diameter'], loc='upper right')
        a1.set_title('(a)', loc='left')
        l3, = a2.plot(MAP.x, MAP.rich, 'o-', c='k'); a2.set_ylim(0.470, 0.525)
        a2.set_xlabel('CO$_2$ at amine inlet (%, dry)'); a2.set_ylabel('Rich loading (mol mol$^{-1}$)')
        t2 = a2.twinx(); t2.spines['right'].set_visible(True)
        l4, = t2.plot(MAP.x, MAP.Trich, 's--', c='#D55E00'); t2.set_ylim(36, 62)
        t2.set_ylabel('Rich solvent temperature (°C)', color='#D55E00'); t2.tick_params(axis='y', colors='#D55E00')
        a2.legend([l3, l4], ['Rich loading', 'Rich solvent temperature'], loc='upper left')
        a2.set_title('(b)', loc='left')
        for a in (a1, a2): a.set_xlim(0, 42)
        fig.tight_layout(); save(fig, 'Fig2_amine_map')

    # Fig. 3  membrane stage
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True); w = 0.26
    for i, f in enumerate(FEED_Y):
        s = B[B.feed == f].set_index(['pf', 'pp']).loc[PRESSURES]
        a1.bar(xs + (i - 1) * w, s.PurB, w, color=C[f], label=f, edgecolor='white', lw=0.4)
        a2.bar(xs + (i - 1) * w, s.AreaB_Mm2, w, color=C[f], edgecolor='white', lw=0.4)
    a1.set_ylabel('Permeate CO$_2$ to amine (%, dry)'); a1.set_ylim(0, 65)
    a2.set_ylabel('Membrane area (10$^6$ m$^2$ per train)'); a2.set_yscale('log')
    for a in (a1, a2):
        a.set_xticks(xs); a.set_xticklabels(CFG_LAB); a.set_xlabel('Feed / permeate pressure (bar)')
    fig.legend(*a1.get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.09))
    letter(a1, 'a'); letter(a2, 'b'); save(fig, 'Fig3_membrane_stage')

    # Fig. 4  pure-gas design error
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    dx = {'NGCC': -0.08, 'Coal': 0.0, 'Cement': 0.08}
    for f in FEED_Y:
        g = df[df.feed == f].groupby(['pf', 'pp']); m_ = B[B.feed == f].set_index(['pf', 'pp']).loc[PRESSURES]
        for a, col in [(a1, 'Under_pct'), (a2, 'CapIfA')]:
            lo = g[col].min().loc[PRESSURES].values; hi = g[col].max().loc[PRESSURES].values; mv = m_[col].values
            a.errorbar(xs + dx[f], mv, yerr=[mv - lo, hi - mv], fmt=MK[f] + '-', c=C[f], capsize=2,
                       lw=1.1, elinewidth=0.8, label=f)
    a1.axhline(0, c='k', lw=0.6); a1.set_ylabel('Membrane area error of pure-gas design (%)')
    a2.axhline(90, c='k', lw=0.8, ls=':'); a2.text(3.35, 90.05, '90 % target', ha='right', va='bottom', fontsize=6.5)
    a2.set_ylabel('Overall CO$_2$ capture achieved (%)'); a2.set_ylim(88.4, 90.6)
    for a in (a1, a2):
        a.set_xticks(xs); a.set_xticklabels(CFG_LAB); a.set_xlabel('Feed / permeate pressure (bar)'); a.set_xlim(-0.4, 3.4)
    fig.legend(*a1.get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.16),
               title='Markers: F·K = 0.191;  bars: F·K = 0.123–0.272', title_fontsize=6.5)
    letter(a1, 'a', x=-0.20, y=1.02); letter(a2, 'b', x=-0.20, y=1.02); save(fig, 'Fig4_design_error')

    # Fig. 5  cost breakdown
    CCOL = ['#BDBDBD', '#E69F00', '#56B4E9', '#CC79A7', '#0072B2']
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    for a, f, L in zip(axs, FEED_Y, 'abc'):
        totals = []
        for x, bdn in enumerate([breakdown(f, None, BASE, False), breakdown(f, best[f], BASE)]):
            bottom = 0
            for k, cn in enumerate(COMP):
                a.bar(x, bdn[cn], 0.6, bottom=bottom, color=CCOL[k], edgecolor='white', lw=0.4,
                      label=cn if f == 'NGCC' and x == 1 else None); bottom += bdn[cn]
            totals.append(bottom)
        top = max(totals)
        for x, tv in enumerate(totals): a.text(x, tv + 0.02 * top, f'{tv:.0f}', ha='center', va='bottom', fontsize=7)
        a.set_ylim(0, top * 1.15); a.set_xticks([0, 1]); a.set_xticklabels(['MEA-only', 'Hybrid']); a.set_xlim(-0.6, 1.6)
        a.set_title(f'{f}  (hybrid: {best[f].pf:g} / {best[f].pp:g} bar)', fontsize=7.5); letter(a, L, x=-0.22)
    axs[0].set_ylabel(f'Cost of compared items ({UNIT})')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15))
    save(fig, 'Fig5_cost_breakdown')

    # Fig. 6  sensitivity tornado (cement)
    LAB = dict(th='Steam-to-power factor (0.15–0.35)', elec='Electricity price (30–120 USD MWh$^{-1}$)',
               rot='Machinery cost (500–1500 USD kW$^{-1}$)', abs='Absorber cost (0.5–3×)',
               mem='Membrane cost (0–100 USD m$^{-2}$)', life='Membrane lifetime (3–10 y)',
               crf='Capital recovery factor (6–12 %)', eta_c='Compressor efficiency (0.70–0.88)',
               eta_v='Vacuum pump efficiency (0.60–0.80)', eta_e='Expander efficiency (0.75–0.90)',
               srd_hyb='Hybrid SRD (±5 %)')
    f = 'Cement'; r = best[f]; base_v = dcost(f, r, BASE); bars = []
    for k, (lo, hi) in RANGES.items():
        v = sorted([dcost(f, r, {**BASE, k: lo}), dcost(f, r, {**BASE, k: hi})]); bars.append((LAB[k], *v))
    bars.sort(key=lambda b_: b_[2] - b_[1])
    fig, ax = plt.subplots(figsize=(5.6, 3.0), constrained_layout=True)
    for i, (lab, lo, hi) in enumerate(bars):
        ax.barh(i, base_v - lo, left=lo, color='#56B4E9', height=0.6)
        ax.barh(i, hi - base_v, left=base_v, color='#E69F00', height=0.6)
    bl = ax.axvline(base_v, c='k', lw=0.8)
    ax.set_yticks(range(len(bars))); ax.set_yticklabels([b_[0] for b_ in bars])
    ax.set_xlim(min(b_[1] for b_ in bars) * 0.9, max(b_[2] for b_ in bars) * 1.05)
    ax.set_xlabel(f'Cost difference, hybrid minus MEA-only ({UNIT})')
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc='#56B4E9'), plt.Rectangle((0, 0), 1, 1, fc='#E69F00'), bl],
              ['Lower end\nof range', 'Upper end\nof range', f'Base case\n({base_v:.1f})'],
              loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=5.5, handlelength=1.2, labelspacing=0.9)
    save(fig, 'Fig6_sensitivity_cement')
    shutil.make_archive('figures', 'zip', 'figures'); print('\nStage 5  figures saved in ./figures and figures.zip')

# =============================================================================
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='smoke test: one source, one pressure, F.K = 0.191')
    ap.add_argument('--from-csv', action='store_true', help='skip Stage 1 and reuse results/stage1_hybrid_design.csv')
    args, _ = ap.parse_known_args()
    stage0()
    if args.from_csv:
        df = pd.read_csv('results/stage1_hybrid_design.csv')
    elif args.quick:
        df = stage1(['Cement'], [(10.0, 1.0)], [0.191])
    else:
        df = stage1(list(FEED_Y), PRESSURES, FKS)
    stage2(df)
    if not args.quick:
        stage3(df); stage5(df)
    stage4()
