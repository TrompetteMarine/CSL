"""Build typed, auditable JSON inputs for artifact-tool source-data workbooks.

No statistical estimates or simulations are recomputed here. Canonical estimates
come from statistics/, and plot values come from figures/source_data/.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
STAT=ROOT/'statistics'
FIG=ROOT/'figures'
pop=pd.read_csv(STAT/'population_outcomes.csv')
batch=pd.read_csv(STAT/'batch_outcomes.csv')
interval=pd.read_csv(STAT/'endpoint_intervals.csv')
wdpop=pd.read_csv(STAT/'withdrawal_population_outcomes.csv')
wdbatch=pd.read_csv(STAT/'withdrawal_batch_outcomes.csv')
wdinterval=pd.read_csv(STAT/'withdrawal_intervals.csv')
trajectory_pop=pd.read_csv(STAT/'trajectory_population_values.csv')
trajectory_batch=pd.read_csv(STAT/'trajectory_batch_means.csv')
trajectory_intervals=pd.read_csv(STAT/'trajectory_intervals.csv')
contrasts=pd.read_csv(STAT/'paired_contrasts.csv')
contrast_pop=pd.read_csv(STAT/'paired_contrast_populations.csv')
contrast_batch=pd.read_csv(STAT/'paired_contrast_batches.csv')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rel(path):return str(path.relative_to(ROOT))

def clean(value):
    if pd.isna(value):return None
    if isinstance(value,(bool,str)):return value
    if isinstance(value,(int,float)):
        if not math.isfinite(value):raise ValueError(f'Non-finite input: {value}')
        return value
    if hasattr(value,'item'):return clean(value.item())
    return str(value)

def sheet(name,df,source,title=None):
    assert len(df)>0,(name,source)
    df=df.reset_index(drop=True).copy()
    # Excel has no numeric infinity. Preserve a finite numeric column and an
    # explicit limit flag, rather than exporting an Excel error or a fake zero.
    for name_col in list(df.select_dtypes(include=['number']).columns):
        mask=np.isinf(df[name_col])
        if mask.any():
            assert name_col=='conditional_df' and (df.loc[mask,name_col]>0).all()
            df[name_col+'_infinite_limit']=mask
            df.loc[mask,name_col]=np.nan
    header_map={'N':'N_total','n':'n_per_community'}
    return dict(name=name,title=title or name,source='Source: '+source,
                headers=[header_map.get(c,c) for c in df.columns],rows=[[clean(v) for v in row] for row in df.itertuples(index=False,name=None)])

def sub(frame,key):
    p=frame
    if key=='F3':return p[(p.group=='two')&(p.initial=='misleading_random')]
    if key=='F4':return p[p.group=='primary']
    if key=='F5':return p[(p.group=='primary')&(p.n==100)&(p.initial=='misleading_random')&p.geometry.isin(['ring','ws0.1','broadcast'])]
    if key=='F6':return p[p.group=='M10000']
    if key=='S1':return p[((p.group=='primary')&(p.n==100)&(p.initial=='misleading_random')&p.geometry.isin(['ring','ws0.01','ws0.1','ws1']))|((p.group=='M10000')&(p.n==50)&(p.initial=='misleading_contiguous')&p.geometry.isin(['ring','ws0.01','ws0.1']))]
    if key=='S2':return p[p.group=='controls']
    if key=='S3':return p[p.group.isin(['primary','followup'])&(p.initial=='misleading_contiguous')&p.n.isin([50,100,200])&p.geometry.isin(['ring','ws0.01','ws0.1'])]
    if key=='S4':return p[((p.group.isin(['primary','choiceonly']))&p.n.isin([50,100])&(p.initial=='misleading_contiguous')&p.geometry.isin(['ws0.01','ws0.1','ws1','regular']))|((p.group.isin(['primary','degree']))&p.n.isin([20,100])&(p.initial=='misleading_contiguous')&p.geometry.isin(['ring','ws0.01','ws0.1','ws1']))]
    if key=='S5':return p[p.group=='scaling']
    if key=='S6':return p[p.group=='two']
    if key=='S7':return p[(p.group=='M10000')&(p.initial=='misleading_contiguous')&p.n.isin([20,50])&p.geometry.isin(['ring','ws0.01','ws0.1'])]
    if key=='S8':return p.iloc[:0]
    raise KeyError(key)

dictionary=[
 ['label','Saved simulation configuration identifier','identifier','Links to results/{label}.npz and metadata JSON'],
 ['graph_id','Graph and initialisation seed batch','identifier','Primary independent Monte Carlo unit'],
 ['rep','Population innovation replicate within a seed batch','identifier','Not an agent or a trial'],
 ['group','Simulation cohort','category','See cohort notes and Supplementary Methods'],
 ['N','Total agents in one population','agents','Design constant'],
 ['n','Agents per community','agents/community','Design constant'],
 ['M','Number of communities','communities','Design constant'],
 ['geometry','Community exposure graph or broadcast condition','category','ring; ws0.01; ws0.1; ws1; regular; broadcast'],
 ['initial','Initial score and signal placement condition','category','misleading_random = scattered; misleading_contiguous = clustered'],
 ['gamma','Between-community mixing weight Γ','weight','B=(1−Γ)I+ΓD⁻¹A; special two-community B stated in Methods'],
 ['lam','Social choice coefficient λ','score units','Withdrawal compares λ=0.6 with λ=0'],
 ['theta','Broadcast mixture θ in choice exposure','weight','Cθ=(1−θ)B+θ1ωᵀ'],
 ['k','Initial ring degree','neighbours/community','Watts–Strogatz construction parameter'],
 ['T','Simulation horizon','trials','Fresh studies 500 or 1,000; continuation 1,000'],
 ['mode','Decisiveness control code','integer','0 primary score; 1 constant 0.5; 2 emitted score 1 with private/peer score 0.5'],
 ['optimal_fraction','Realised better-arm choice fraction','fraction','Mean over final 100 trials for endpoint tables'],
 ['private_correct','Agents with terminal Q(0)>Q(1)','fraction','Terminal score ranking; differs from private_prechoice path metric'],
 ['private_prechoice','Agents with Q(0)>Q(1) before a trial','fraction','For trajectory tables, trailing ten-trial mean within population'],
 ['private_gap','Terminal population mean of Q(0)−Q(1)','score units','Arithmetic mean computed from stored score arrays; source storage precision is recorded separately'],
 ['latent_public_gap','Terminal private_correct minus final-window optimal_fraction','fraction-point difference','Not same-time individual disagreement'],
 ['mean_removed_probability','Choice probability if contemporaneous social input were set to zero','probability','Mean of saved per-trial readout over final 100 trials'],
 ['regret','Expected regret from realised choices and fixed reward means','reward units/agent','(μ0−μ1) times the cumulative worse-arm choice fraction'],
 ['wrong','Population endpoint choice fraction below 0.5','indicator','One dependent population outcome; not a binomial agent count'],
 ['wrong_communities','Communities with final-window better-arm choice fraction below 0.5','fraction','Within-population descriptive outcome'],
 ['community_variance','Variance of community final-window choice fractions','fraction squared','Within-population descriptive population variance'],
 ['clustering','Mean local clustering of the community graph','coefficient','Graph quantity, repeated within innovation replicates'],
 ['mean_path','Mean shortest-path length of the community graph','edges','Computed on connected simulated graphs'],
 ['degree_sd','Standard deviation of community graph degrees','neighbours','Graph quantity, not sampling standard error'],
 ['first_step_probability','Mean better-arm choice probability immediately after withdrawal decision','probability','Identical saved entry states in both paired conditions'],
 ['mean','Equal-weight mean of seed-batch means','metric units','Balanced innovation counts in each cohort'],
 ['value','Population or seed-batch outcome identified by its table','metric units','Units given by metric and table title'],
 ['effect','Intervention minus reference mean','metric units','Paired contrast; covariance retained'],
 ['mcse','Monte Carlo standard error across seed-batch means','metric units','sqrt(sample variance of batch means / B)'],
 ['ci95_low / ci95_high','Untruncated approximate 95% seed-batch t Monte Carlo interval','metric units','mean ± t(.975,B−1) × mcse; pointwise for curves'],
 ['df','Degrees of freedom for marginal seed-batch t interval','degrees of freedom','B−1'],
 ['batches','Number B of graph/initialisation seed batches','seed batches','B=4 main, B=16 expanded, B=2 many-community'],
 ['populations','Population trajectories contributing to an estimate','populations','Typically 4 innovations/batch; many-community 2'],
 ['conditional_mcse','Innovation-only Monte Carlo standard error conditional on realised seed batches','metric units','Different target from marginal mcse; do not add or substitute'],
 ['conditional_df','Welch–Satterthwaite degrees of freedom for conditional interval','degrees of freedom','Different target from marginal df'],
 ['conditional_df_infinite_limit','Conditional degrees of freedom have their infinite limit','boolean','If true, conditional_df is blank because Excel has no numeric infinity; the source CSV stores +inf'],
 ['conditional_ci95_low / conditional_ci95_high','Conditional innovation-only Monte Carlo interval','metric units','See Supplementary Methods; realised graph/initialisation batches held fixed'],
 ['variance','Estimated variance of a seed-batch mean estimator','metric units squared','mcse squared'],
 ['covariance_of_condition_means','Estimated covariance of paired condition mean estimators','metric units squared','Used in Var(Y−X)=Var(Y)+Var(X)−2Cov(X,Y)'],
 ['hoeffding95_low / hoeffding95_high','Distribution-free bounded-outcome 95% Monte Carlo limits','metric units','Untruncated Hoeffding bound, provided as a conservative comparison'],
 ['hoeffding95_support_low / hoeffding95_support_high','Hoeffding limits intersected with known outcome support','metric units','Distinct from untruncated approximate t intervals'],
 ['trial','One-based simulation trial','trial','Curves use trailing ten trials, including partial windows at trials 1–9'],
 ['curve_id','Trajectory cohort identifier','identifier','Complete cohort mapping in Readme'],
 ['source_dtype','Saved terminal score floating-point storage','category','Readout precision audited separately; not a Monte Carlo interval'],
 ['panel','Lower-case panel label within the indicated figure','identifier','Cross-references the figure legend'],
 ['p','Better-arm choice probability','probability','Analytic grid for the score functions'],
 ['d','Decisiveness score supplied to a private gain function','score','Primary model realises d≥0.5; full [0,1] interval is drawn for comparison'],
 ['x','Private score gap plus social choice input','score units','Analytic response uses p=logistic(12x)'],
 ['primary','Primary decisiveness score or its signed cue response','score / signed score','s(p)=max(p,1−p); response=s(p)(2p−1), according to the grid title'],
 ['constant_half','Constant score 0.5 or its signed response','score / signed score','Response=0.5(2p−1) in the response grid'],
 ['quadratic_floor_half','Quadratic score with floor 0.5','score','0.5+0.5(2p−1)²'],
 ['zero_floor','Absolute-distance score or its signed response','score / signed score','s(p)=|2p−1|; response=s(p)(2p−1) in the response grid'],
 ['alpha_plus / alpha_minus','Private reward-conditional learning gains','gain','alpha_plus=0.4−0.35d; alpha_minus=0.05+0.35d'],
 ['common_multiplier / difference_multiplier','Frozen-score two-community local derivative multipliers','dimensionless','1.8 and 1.8(1−2Γ), respectively; full-process learning held fixed'],
 ['bin_left / bin_right / bin_centre','Histogram bin edges and centre','better-arm choice fraction','25 bins on [0,1]; final bin includes its right endpoint'],
 ['community_count','Communities in an indicated within-population histogram bin','communities','Denominator 10,000 per population; these are dependent outcomes'],
 ['fraction','Within-population histogram fraction, or its average as identified by table','fraction','community_count divided by the number of communities'],
 ['state_id','Fixed saved entry-state identifier','identifier','g{graph_id}_r{rep}; 16 states nested in four seed batches'],
 ['routing','Counterfactual routing matrix for the same saved sender state','category','ring or broadcast; not a new simulated trajectory'],
 ['receiving_community / other_community','Receiver indices in a conditional field correlation matrix','one-based community index','100 communities in each saved sender population'],
 ['mean_conditional_correlation','Entrywise average of exact fixed-state conditional field correlations','correlation coefficient','Average over 16 sender states; not a sample correlation from 16 observations'],
 ['entry_states','Saved sender states used in the conditional-noise ensemble','states','Four seed batches × four population innovation replicates'],
 ['mean_better_propensity','Population mean choice probability at a fixed sender state','probability','Conditional on all saved private scores and lagged signals'],
 ['conditional_public_sd','Exact conditional SD of the next population choice fraction','fraction','sqrt(sum_i p_i(1−p_i))/N; independent current action draws conditional on saved state'],
 ['signal_variance / signal_sd','Exact conditional variance or SD of the next signed community signal','score squared / score','At the fixed saved sender state; independent of Monte Carlo estimation uncertainty'],
 ['expected_signal','Expected next signed community signal conditional on saved state','signed score','Community mean of d_i(2p_i−1)'],
 ['community_signal_rms_sd','Root-mean-square conditional SD over sender-community signals','score','sqrt(mean of the diagonal conditional signal variances)'],
 ['broadcast_field_sd','Exact conditional SD of the common broadcast field','score','Same field enters every receiving community; conditional covariance rank one'],
 ['ring_field_rms_sd','Root-mean-square receiver SD under counterfactual ring routing','score','sqrt(mean diagonal of C_ring Cov(signal) C_ringᵀ)'],
 ['broadcast_covariance_rank / ring_covariance_rank','Rank of the exact conditional receiver-field covariance matrix','rank','Computed at each fixed sender state under the indicated routing'],
 ['broadcast_offdiagonal_correlation','Conditional field correlation between distinct broadcast receivers','correlation coefficient','One whenever the common broadcast field has positive variance'],
 ['ring_offdiagonal_correlation_mean','Mean conditional correlation over distinct ring receivers','correlation coefficient','Descriptive receiver-pair average within the fixed sender state'],
 ['ring_to_broadcast_field_rms_sd_ratio','Ring field RMS conditional SD divided by broadcast field SD','ratio','Descriptive exact fixed-state comparison'],
 ['public_broadcast_field_correlation','Exact conditional correlation of public better-arm fraction and broadcast signal','correlation coefficient','Shared current action draws produce this covariance'],
 ['local_derivative_lambda','Derivative of the current mean choice probability with respect to λ','probability per unit λ','Saved scores and lagged signals held fixed'],
 ['exact_change','Withdrawn probability minus retained probability at an identical entry state','probability difference','Exact nonlinear readout, λ=0 compared with λ=0.6'],
 ['first_order_prediction_for_lambda_change','Local derivative times the finite change −0.6 in λ','probability difference','Diagnostic linear approximation; does not replace the exact nonlinear contrast'],
 ['actual_float32_readout_change_retained / actual_float32_readout_change_withdrawn','Readout change caused by rounding the saved private scores to float32','probability difference','Readout sensitivity only; not a global simulator forward-error bound'],
]

mapping={
 'F3':('Fig_3_Source_Data.xlsx','Figure 3 — Two-community decisions and local spectrum','main02_two_communities'),
 'F4':('Fig_4_Source_Data.xlsx','Figure 4 — Community size and network geometry','main03_size_geometry'),
 'F5':('Fig_5_Source_Data.xlsx','Figure 5 — Private rankings and influence withdrawal','main04_private_withdrawal'),
 'F6':('Fig_6_Source_Data.xlsx','Figure 6 — Ten thousand communities','main05_many_communities'),
 'S2':('Supplementary_Fig_2_Source_Data.xlsx','Supplementary Figure 2 — Community graph diagnostics','supp01_graph_diagnostics'),
 'S6':('Supplementary_Fig_6_Source_Data.xlsx','Supplementary Figure 6 — Initialisation and score controls','supp02_controls'),
 'S3':('Supplementary_Fig_3_Source_Data.xlsx','Supplementary Figure 3 — Expanded geometry replications','supp03_expanded_geometry'),
 'S4':('Supplementary_Fig_4_Source_Data.xlsx','Supplementary Figure 4 — Exposure channels and degree','supp04_choice_degree'),
 'S5':('Supplementary_Fig_5_Source_Data.xlsx','Supplementary Figure 5 — Population scaling','supp05_scaling'),
 'S1':('Supplementary_Fig_1_Source_Data.xlsx','Supplementary Figure 1 — Two-community initialisations','supp06_two_initialisations'),
 'S7':('Supplementary_Fig_7_Source_Data.xlsx','Supplementary Figure 7 — Community outcome distributions','supp07_many_detail'),
 'S8':('Supplementary_Fig_8_Source_Data.xlsx','Supplementary Figure 8 — Score functions and learning gains','supp08_score_response'),
 'S10':('Supplementary_Fig_10_Source_Data.xlsx','Supplementary Figure 10 — Conditional signal-noise propagation','supp10_noise_propagation'),
}

notes_common=[
 ['Data type','Simulated population outcomes from frozen records; no human participant measurements and no new simulations in this export.'],
 ['Excel header crosswalk','Native Excel table headers are case-insensitive. To distinguish total population size N from community size n, the workbooks use N_total and n_per_community. Original CSV column names remain N and n.'],
 ['Independent unit','Graph and initialisation seed batch. Agents, communities and trials within a population are dependent. Innovation replicates are nested within a seed batch.'],
 ['Pairing','Innovation indices are paired across comparison conditions where stated. Contrasts preserve pairing and the full covariance; conditions are not treated as unrelated samples.'],
 ['Endpoint window','Choice fractions and community outcomes use the final 100 trials. Private score rankings use terminal saved Q. Fresh horizons are T=1,000 except scaling and M=10,000 studies, where T=500.'],
 ['95% MC intervals','Untruncated approximate Student-t limits across independent seed-batch means: mean ± t(.975,B−1) × MCSE. Small-B calibration is unvalidated. These are not uncertainty intervals about human behaviour or model validity.'],
 ['Conditional target','Conditional MCSE holds the realised graph and initialisation batches fixed and measures innovation uncertainty only. It is a separate estimand, not an additional error term to add to marginal MCSE.'],
 ['Trajectories','Ten-trial trailing averages are formed within each population before averaging within seed batches. Trials 1–9 use the available prefix. Trajectory intervals are pointwise, not simultaneous bands.'],
 ['Precision','Underlying numerical values are retained at available source precision; display rounding does not alter them. Blank means inapplicable/not supplied, never zero. Excel cannot store numeric infinity: conditional_df is blank and conditional_df_infinite_limit=true in that case. The source CSV retains +inf.'],
 ['Inference scope','No null-hypothesis tests or p-values are used. Model parameters are fixed design constants. Monte Carlo uncertainty, finite-size/theory error and floating-point readout sensitivity are distinct.'],
 ['Reproduction','Run statistics/analyse_uncertainty.py and figures/rebuild_figures.py from the complete source package, then source_data/prepare_workbook_inputs.py and source_data/build_workbooks.mjs. Raw innovation trajectories are recoverable from the provided seeds and engine.'],
]

manifest_path=FIG/'Source_Data_Manifest.json'
if not manifest_path.exists():
    raise SystemExit('Waiting for figures/Source_Data_Manifest.json; no workbook inputs written.')
manifest=json.loads(manifest_path.read_text())

# Accommodate the figure export manifest while keeping its full content in the
# resulting audit ledger. Adapt this small function if its explicit schema changes.
def manifest_entries(figure):
    if isinstance(manifest,dict):
        entries=manifest.get('figures',manifest)
        if isinstance(entries,dict):
            v=entries.get(figure,[])
            if isinstance(v,dict):
                return v.get('files',v.get('sources',v.get('tables',[])))
            return v
        return [e for e in entries if e.get('figure')==figure]
    return [e for e in manifest if e.get('figure')==figure]

workbooks=[];indices=[];sources={}
sheet_names={
 'population_values':'Population values','batch_means':'Batch means','plotted_intervals':'Plotted intervals',
 'trajectory_population_values':'Population curves','trajectory_batch_means':'Batch curves','trajectory_intervals':'Plotted trajectory intervals',
 'analytic_multipliers':'Frozen multipliers','annotated_multiplier':'Annotated multiplier',
 'paired_withdrawal_populations':'Paired withdrawal populations','withdrawal_batch_means':'Withdrawal batches','withdrawal_intervals':'Withdrawal intervals',
 'population_bin_counts':'Population bin counts','batch_histograms':'Batch histograms','plotted_mean_histograms':'Plotted mean histograms',
 'score_grid':'Score grid','response_grid':'Response grid','gain_grid':'Gain grid',
 'plotted_mean_correlations':'Mean conditional correlations','conditional_noise_population':'Conditional noise populations',
 'conditional_noise_batch':'Conditional noise batches','conditional_noise_ensemble_summary':'Conditional noise summary',
}
for key,(filename,title,figure) in mapping.items():
    sheets=[];paths=[]
    for j,entry in enumerate(manifest_entries(figure)):
        entry={'path':entry} if isinstance(entry,str) else entry
        p=Path(entry.get('path',entry.get('file',entry.get('csv',''))))
        if not p.is_absolute():
            p=ROOT/p if (ROOT/p).exists() else FIG/p
        frame=pd.read_csv(p)
        suffix=p.stem.removeprefix(figure+'_')
        name=entry.get('sheet',entry.get('name',sheet_names.get(suffix,suffix.replace('_',' ').title())))[:31]
        if any(x['name']==name for x in sheets):name=f'Plot values {j+1}'
        sheets.append(sheet(name,frame,rel(p),entry.get('title',name)))
        paths.append(p)
    if not sheets:raise ValueError(f'No plotted source data in manifest for {figure}')
    if key=='F5':
        for name,frame,p in [('Paired contrasts',contrasts,STAT/'paired_contrasts.csv'),('Paired population values',contrast_pop,STAT/'paired_contrast_populations.csv'),('Paired batch values',contrast_batch,STAT/'paired_contrast_batches.csv')]:
            relevant=frame[frame.contrast_id.str.contains('withdraw')]
            sheets.append(sheet(name,relevant,rel(p)));paths.append(p)
    if key=='S10':
        for name,filename_extra in [('Community noise values','conditional_noise_community.csv'),('Withdrawal sensitivity','withdrawal_functional_sensitivity.csv')]:
            p=STAT/filename_extra
            sheets.append(sheet(name,pd.read_csv(p),rel(p)));paths.append(p)
    sheets.append(dict(name='Variable definitions',title='Definitions and units',source='Source: Supplementary Methods and simulation metadata.',headers=['variable','definition','units','scope'],rows=dictionary))
    notes=[['Figure',title],['Plot file','figures/'+figure+'.pdf']]+notes_common
    if key=='S8':
        notes=[['Figure',title],['Plot file','figures/'+figure+'.pdf'],
          ['Data type','Deterministic double-precision evaluations of analytic functions. These are not simulated population outcomes or new experimental observations.'],
          ['Independent unit','Not applicable to this deterministic grid. Grid points are function-evaluation locations, not independent statistical observations.'],
          ['Pairing','Not applicable to this deterministic grid. Functions are evaluated on common axes for mathematical comparison.'],
          ['Endpoint window','Not applicable. Score and response grids each contain 1,001 evaluation points; the private-gain grid contains 200 evaluation points. There is no trial horizon or final-window summary.'],
          ['95% MC intervals','Not applicable. Fixed deterministic function evaluations have no Monte Carlo sampling interval. No error bars or sampling intervals are reported.'],
          ['Conditional target','Not applicable. No conditional or marginal Monte Carlo estimate is calculated for this figure.'],
          ['Trajectories','Not applicable. The curves are static function graphs, not simulated time trajectories.'],
          ['Parameters','The response grid uses h=12. Learning gains use alpha_min=0.05 and alpha_max=0.4: alpha_plus(d)=0.4−0.35d and alpha_minus(d)=0.05+0.35d. The primary score realises d≥0.5; the gain graph shows the full [0,1] domain for comparison.'],
          ['Precision','Numerical grid values are retained at available double precision. Cell display rounding does not change their stored values. Sampling confidence intervals do not apply.'],
          ['Inference scope','Analytic score-function comparisons only. This figure supplies neither an empirical robustness experiment nor estimates of human behaviour.'],
          ['Reproduction','The score_response function in figures/rebuild_figures.py contains the complete analytic definitions and grids. No new simulations are required.']]
    if key=='S10':
        scoped={
          'Data type':'Analytic conditional-noise summaries and readout sensitivities evaluated at 16 saved simulation entry states. The figure contains no new population trajectories or human observations.',
          'Independent unit':'For ensemble Monte Carlo estimates, four independent graph/initialisation seed batches, with four saved states in each batch. Receiving communities and entries of a conditional correlation matrix are not independent statistical replications.',
          'Pairing':'Ring and broadcast routing are evaluated at the same saved sender state. Conditional noise is calculated analytically, rather than estimated from separately sampled action draws.',
          'Endpoint window':'These are fixed entry-state calculations after 1,000 original trials. Conditional covariance refers to the next action draw; withdrawal sensitivity holds the identical saved scores and lagged signals fixed. No final-100-trial window defines these conditional-noise quantities.',
          '95% MC intervals':'When supplied, approximate Student-t intervals quantify uncertainty of an ensemble-mean conditional summary across the four seed batches. They are not the conditional action-noise SD itself. Small-batch interval calibration is unvalidated.',
          'Conditional target':'The action-noise variance and covariance are exact model expressions conditional on one complete sender state. Monte Carlo uncertainty over the saved entry-state ensemble is a separate level of uncertainty.',
          'Trajectories':'Not applicable to the plotted conditional matrices and SDs. No new time path is simulated for this figure.',
          'Inference scope':'Conditional model calculations at the saved entry states. They do not establish a long-time covariance law or sampling uncertainty about human behaviour. Fixed design parameters are not treated as fitted random estimates.',
          'Reproduction':'statistics/conditional_noise.py computes the state-conditional expressions and covariance matrices; statistics/analyse_uncertainty.py supplies the readout sensitivity diagnostics. All calculations use the saved entry-state ensemble.'}
        notes=[[label,scoped.get(label,value)] for label,value in notes]
    if key=='F3':notes.append(['Analytic panel','Local frozen-score multipliers only: λs(0.5)h/2=1.8 for the common mode, and 1.8(1−2Γ) for the difference mode. They are not a full-process bifurcation or accuracy theorem.'])
    if key=='F5':notes.append(['Withdrawal design','All 16 saved broadcast entry populations are continued under retained λ=0.6 and withdrawn λ=0, with paired fresh innovation streams. No outcome-based selection is used.'])
    if key=='F6':notes.append(['Sparse replication','Only two seed batches, each containing two population trajectories per cell. Intervals are especially weakly calibrated; inspect the plotted batch points.'])
    if key=='S7':notes.append(['Histogram denominator','Each population has 10,000 communities. Community bins are dependent descriptive outcomes within a population; they are not 10,000 independent replications.'])
    if key=='S8':notes.append(['Analytic data','Deterministic function evaluations on the exact grid used in the figure. No simulated sample, sampling variance, independent replications or confidence interval applies.'])
    if key=='S10':
        notes.append(['Conditional noise','Exact action-noise covariance is evaluated at each of the 16 fixed saved sender states. Correlation matrices are averaged entrywise over those states. They are not correlations estimated from 16 observations.'])
        notes.append(['SD versus MCSE','Panel d compares the exact conditional field standard deviations: ring_field_rms_sd and broadcast_field_sd. The summary MCSE describes uncertainty in their ensemble mean across four seed batches. It is not the conditional action-noise SD itself.'])
        notes.append(['Routing comparison','Ring and broadcast field covariances use the same fixed actual sender states. Their difference is a counterfactual routing calculation, not evidence from a newly simulated ring population or a long-time covariance theorem.'])
        notes.append(['Panel d display units','The figure plots 1,000 × ring_field_rms_sd or 1,000 × broadcast_field_sd and labels its axis ×10⁻³. Source values remain in natural score units.'])
    for j,entry in enumerate(manifest_entries(figure)):
        if isinstance(entry,dict) and entry.get('description'):
            notes.append([f'Plot source {j+1}',entry['description']])
    notes.append(['Source tables','; '.join(sorted(set(rel(p) for p in paths)))])
    workbooks.append(dict(id=key,file=filename,title=title,notes=notes,sheets=sheets))
    indices.append(dict(figure=key,figure_file=figure+'.pdf',workbook='outputs/nhb_20261016/'+filename,sheets=len(sheets)+1,rows=sum(len(s['rows']) for s in sheets),source_files=len(set(paths))))
    sources[key]=[{ 'path':rel(p),'sha256':sha(p)} for p in sorted(set(paths))]

# One additional workbook reproduces the exact statistical comparison records.
stat_sheets=[]
named_table_paths=[]
for name,filename in [('Main Table 1','Table_1_Source_Data.csv'),('Main Table 2','Table_2_Source_Data.csv'),('Geometry contrasts','Supplement_Table_Geometry.csv')]:
    p=STAT/filename
    if p.exists():
        stat_sheets.append(sheet(name,pd.read_csv(p),rel(p)))
        named_table_paths.append(p)
stat_sheets += [sheet('Endpoint estimates',interval,rel(STAT/'endpoint_intervals.csv')),
             sheet('Paired contrasts',contrasts,rel(STAT/'paired_contrasts.csv')),
             sheet('Paired batch values',contrast_batch,rel(STAT/'paired_contrast_batches.csv')),
             sheet('Paired population values',contrast_pop,rel(STAT/'paired_contrast_populations.csv')),
             dict(name='Variable definitions',title='Definitions and units',source='Source: Supplementary Methods and statistics analysis.',headers=['variable','definition','units','scope'],rows=dictionary)]
workbooks.append(dict(id='Tables',file='Statistical_Comparisons_Source_Data.xlsx',title='Statistical comparisons — source data',notes=[['Content','Exact endpoint estimates and paired contrasts, nested raw paired population values and seed-batch means. Main and supplementary statistical tables select rows from this canonical record.'],['Display units','Source probabilities and probability differences remain fractions. The manuscript displays probabilities multiplied by 100 as percentages, and differences/MCSEs multiplied by 100 as percentage points. Regret remains in payoff units.']]+notes_common,sheets=stat_sheets))
indices.append(dict(figure='Statistical tables',figure_file='',workbook='outputs/nhb_20261016/Statistical_Comparisons_Source_Data.xlsx',sheets=len(stat_sheets)+1,rows=sum(len(s['rows']) for s in stat_sheets),source_files=4))
table_paths=[STAT/'endpoint_intervals.csv',STAT/'paired_contrasts.csv',STAT/'paired_contrast_batches.csv',STAT/'paired_contrast_populations.csv']+named_table_paths
sources['Tables']=[{'path':rel(p),'sha256':sha(p)} for p in table_paths]
indices[-1]['source_files']=len(table_paths)
def output_order(key):
    if key.startswith('F') and key[1:].isdigit():return (0,int(key[1:]))
    if key.startswith('S') and key[1:].isdigit():return (1,int(key[1:]))
    return (2,0)
workbooks.sort(key=lambda w:output_order(w['id']))
indices.sort(key=lambda item:output_order(item['figure']))
(HERE/'workbook_inputs.json').write_text(json.dumps(dict(workbooks=workbooks),allow_nan=False,separators=(',',':')))
pd.DataFrame(indices).to_csv(HERE/'Source_Data_Index.csv',index=False)
(HERE/'Source_Hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps(dict(workbooks=len(workbooks),rows=sum(i['rows'] for i in indices),json_bytes=(HERE/'workbook_inputs.json').stat().st_size),indent=2))
