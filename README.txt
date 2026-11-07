Idea 2: Reference Selection (Seamless Interaction, Naturalistic train)

Data
  data/0000_00NN/  npz + json of archives 0000/0000-0009 (video and audio not downloaded)

Run (python with numpy, scipy, matplotlib)
  python scripts/download.py data 0 0 1 2 3 4 5 6 7 8 9
  python scripts/analysis1.py data results/extended_a0-9 "0000_000[0-9]"
  python scripts/analysis1.py data results/primary_a0-2 "0000_000[0-2]"
  python scripts/analysis2.py data results/idea2_a0-9 "0000_000[0-9]"
  python scripts/make_figure_idea2.py results/idea2_a0-9/analysis2_results.json figures/fig_idea2
  (on Bridges-2: sbatch scripts/analysis2.slurm runs analysis2.py, about 3 minutes)

Outputs
  report/idea2_paper_text.tex                 Idea 2 text, figure and caption for the paper
  figures/fig_idea2.pdf                       the Idea 2 figure (copy in report/figures/)
  results/extended_a0-9/analysis1_results.json   1 x 10 s vs 2 x 5 s comparison (key "refsel"), archives 0-9
  results/primary_a0-2/analysis1_results.json    same, three proposal archives only
  results/idea2_a0-9/analysis2_results.json      fixed-duration, duration, selection, saturation, low-quality conditions
  results/v1_singlefit/                       old run with a different scaling (superseded)

Optional figures (deleted, can be regenerated)
  python scripts/make_figure.py results/extended_a0-9/analysis1_results.json figures/fig1_extended
  python scripts/make_figure2.py results/idea2_a0-9/analysis2_results.json figures/fig2_idea2

Midterm (Section 6, Idea 2): learned and frozen reference encoders on held-out participants
  Extra training-only data: batches 0001-0005, archives 0-9 (new participants)
    sbatch scripts/download_batches.slurm
  Features + 4-fold training/evaluation (one V100, about 25 minutes)
    sbatch scripts/idea2_midterm.slurm "000[0-5]_000[0-9]" results/idea2_midterm 4000
    (runs scripts/idea2_features.py data cache ... then scripts/idea2_midterm.py cache data ...)
  Figure
    python scripts/make_figure_idea2_midterm_v2.py results/idea2_midterm/idea2_midterm_results.json figures/idea2_figure
  Cross-session posture probe (P0047, P0048)
    python scripts/idea2_session_probe.py cache results/idea2_midterm/session_probe.json
  Outputs
    report/idea2_midterm_section.tex          Section 6 text, table and two figures
    figures/idea2_figure1.pdf                 Section 6 Figure 1, contaminated references (copy in report/figures/)
    figures/idea2_figure2.pdf                 Section 6 Figure 2, per-participant accuracy, two-session participants shaded (copy in report/figures/)
    results/idea2_midterm/idea2_midterm_results.json   all conditions, paired differences, attention weights
  Note: archives 0000_0010-0000_0044 contain the SAME 32 participants as 0000_0000-0009,
  so they must not be used as training-only data (they are on disk but not used).
