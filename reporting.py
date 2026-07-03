"""
reporting.py
Step 4 – Reporting + all standout requirements:
  #1  PDF report with confusion matrix, summary stats, diagnostics
  #2  Time trend charts (latency + missing %) from DB history
  #4  Evidently monitoring results included in PDF
"""
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json, os
from datetime import datetime
from typing import Optional

# PDF support (Standout #1)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Image as RLImage, Table, TableStyle,
                                    HRFlowable, PageBreak)
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path  = config['output_folder_path']
model_path        = config['output_model_path']
test_data_path    = config['test_data_path']
prod_path         = config['prod_deployment_path']


# ─────────────────────────────────────────────────────────────────────────────
# Core reporting function (required by rubric)
# ─────────────────────────────────────────────────────────────────────────────
def score_model():
    """Generate confusionmatrix.png using test data and deployed model."""
    import diagnostics
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_true  = test_df['exited']
    predictions = diagnostics.model_predictions(test_df)

    cm = metrics.confusion_matrix(y_true, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['Actual 0', 'Actual 1'], ax=ax)
    ax.set_title('Confusion Matrix – Attrition Risk Model')
    plt.tight_layout()

    os.makedirs(model_path, exist_ok=True)
    cm_path = os.path.join(model_path, 'confusionmatrix.png')
    plt.savefig(cm_path, dpi=100)
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")
    return cm_path


# ─────────────────────────────────────────────────────────────────────────────
# Standout #2 – Time trend charts
# ─────────────────────────────────────────────────────────────────────────────
def _generate_trend_charts(output_dir: str) -> dict:
    """
    Pull diagnostics history from SQL DB (Standout #3 integration)
    and build time-trend line charts for:
      - ingestion + training latency over time
      - average missing value % over time
    Falls back to olddiagnostics/ JSON files when DB is empty.
    """
    import dbsetup
    history = dbsetup.get_diagnostics_history()

    # Supplement with olddiagnostics/ JSON snapshots if needed
    import glob
    if len(history) < 2:
        for snap_path in sorted(glob.glob('olddiagnostics/diagnostics_*.json')):
            with open(snap_path) as sf:
                snap = json.load(sf)
            miss_vals = [m['missing_pct'] for m in snap.get('missing_data', [])]
            history.append({
                'recorded_at':    snap.get('timestamp', ''),
                'ingestion_time': snap.get('ingestion_time', 0.0),
                'training_time':  snap.get('training_time', 0.0),
                'avg_missing_pct': np.mean(miss_vals) if miss_vals else 0.0,
            })

    plots = {}
    os.makedirs(output_dir, exist_ok=True)

    if len(history) < 1:
        return plots

    labels         = [h['recorded_at'][:16] for h in history]   # "YYYY-MM-DDTHH:MM"
    ingestion_vals = [h.get('ingestion_time', 0) or 0 for h in history]
    training_vals  = [h.get('training_time', 0)  or 0 for h in history]
    missing_vals   = [h.get('avg_missing_pct', 0) or 0 for h in history]

    x = range(len(labels))

    # Latency trend
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x, ingestion_vals, 'o-', label='Ingestion (s)', color='#3498db')
    ax.plot(x, training_vals,  's-', label='Training (s)',  color='#e67e22')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Latency Trends – Ingestion & Training')
    ax.legend(fontsize=9)
    plt.tight_layout()
    p1 = os.path.join(output_dir, 'trend_latency.png')
    plt.savefig(p1, dpi=100); plt.close()
    plots['trend_latency'] = p1

    # Missing % trend
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x, missing_vals, 'D-', color='#e74c3c', label='Avg Missing %')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Avg Missing Value %')
    ax.set_title('Missing Value % Trend Over Time')
    ax.legend(fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(output_dir, 'trend_missing.png')
    plt.savefig(p2, dpi=100); plt.close()
    plots['trend_missing'] = p2

    return plots


# ─────────────────────────────────────────────────────────────────────────────
# Standout #1 – Full PDF report (+ #2 + #4 embedded)
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report() -> Optional[str]:
    """
    Generate a comprehensive PDF report including:
      - Model metadata (F1 score, ingested files)
      - Confusion matrix plot
      - Summary statistics table
      - Missing data table
      - Evidently monitoring results: drift, classification performance, prediction drift (Standout #4)
      - Time trend charts: latency + missing % (Standout #2)
    """
    if not REPORTLAB_AVAILABLE:
        print("reportlab not installed – skipping PDF. pip install reportlab")
        return None

    import diagnostics
    import evidently_monitoring as ev_mon

    os.makedirs(model_path, exist_ok=True)
    pdf_path = os.path.join(model_path, 'model_report.pdf')
    doc      = SimpleDocTemplate(pdf_path, pagesize=letter,
                                  leftMargin=0.75*inch, rightMargin=0.75*inch,
                                  topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles   = getSampleStyleSheet()
    h1       = styles['Heading1']
    h2       = styles['Heading2']
    normal   = styles['Normal']
    story    = []

    def section(title):
        story.append(Spacer(1, 0.15*inch))
        story.append(HRFlowable(width='100%', thickness=1, color=colors.steelblue))
        story.append(Paragraph(title, h2))
        story.append(Spacer(1, 0.08*inch))

    def img(path, w=6*inch, h=None):
        if os.path.exists(path):
            ri = RLImage(path, width=w, height=h or w*0.55)
            story.append(ri)
            story.append(Spacer(1, 0.1*inch))

    def styled_table(data, col_widths=None):
        t = Table(data, colWidths=col_widths, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1,  0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR',     (0, 0), (-1,  0), colors.whitesmoke),
            ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0), (-1, -1), 8),
            ('GRID',          (0, 0), (-1, -1), 0.4, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.15*inch))

    # ── Cover ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Dynamic Risk Assessment", styles['Title']))
    story.append(Paragraph("Model Monitoring & Performance Report", h1))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal))
    story.append(Spacer(1, 0.2*inch))

    # F1 score
    score_file = os.path.join(prod_path, 'latestscore.txt')
    if os.path.exists(score_file):
        with open(score_file) as f:
            f1_val = f.read().strip()
        story.append(Paragraph(f"<b>Deployed Model F1 Score:</b> {f1_val}", normal))

    # Ingested files
    ingested_txt = os.path.join(prod_path, 'ingestedfiles.txt')
    if os.path.exists(ingested_txt):
        section("Training Data — Ingested Files")
        with open(ingested_txt) as f:
            files = [l.strip() for l in f if l.strip()]
        for fi in files:
            story.append(Paragraph(f"• {fi}", normal))

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    section("Confusion Matrix")
    cm_path = os.path.join(model_path, 'confusionmatrix.png')
    img(cm_path, w=4.5*inch, h=3.5*inch)

    # ── Summary Statistics ───────────────────────────────────────────────────
    section("Summary Statistics")
    stats = diagnostics.dataframe_summary()
    rows  = [['Column', 'Mean', 'Median', 'Std Dev']]
    for s in stats:
        rows.append([s['column'], f"{s['mean']:.4f}", f"{s['median']:.4f}", f"{s['std']:.4f}"])
    styled_table(rows, col_widths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])

    # ── Missing Data ─────────────────────────────────────────────────────────
    section("Missing Data Analysis")
    missing = diagnostics.dataframe_missing()
    rows2   = [['Column', 'Missing %']]
    for m in missing:
        rows2.append([m['column'], f"{m['missing_pct']:.2f}%"])
    styled_table(rows2, col_widths=[3.5*inch, 2*inch])

    # ── Diagnostics timing ───────────────────────────────────────────────────
    section("Execution Timing (Latest Run)")
    try:
        import dbsetup
        hist = dbsetup.get_diagnostics_history()
        if hist:
            last = hist[-1]
            t_rows = [['Task', 'Time (seconds)'],
                      ['Data Ingestion', f"{last.get('ingestion_time', 0):.4f}"],
                      ['Model Training', f"{last.get('training_time', 0):.4f}"]]
            styled_table(t_rows, col_widths=[3*inch, 2*inch])
    except Exception:
        pass

    # ── Evidently: Data Drift (Standout #4) ─────────────────────────────────
    story.append(PageBreak())
    section("Evidently Monitoring — Feature Data Drift")
    ev_results = ev_mon.run_full_monitoring()
    drift       = ev_results['data_drift']
    drift_sum   = drift.get('_summary', {})
    story.append(Paragraph(
        f"Drifted columns: {drift_sum.get('drifted_columns', '?')} / "
        f"{drift_sum.get('total_columns', '?')} — "
        f"Dataset drift detected: <b>{drift_sum.get('dataset_drift_detected', '?')}</b>",
        normal))
    story.append(Spacer(1, 0.1*inch))

    drift_rows = [['Feature', 'KS Statistic', 'p-value', 'Drift Detected']]
    from diagnostics import model_predictions  # noqa
    features_list = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    for col in features_list:
        d = drift.get(col, {})
        drift_rows.append([col,
                           str(d.get('ks_statistic', 'N/A')),
                           str(d.get('p_value', 'N/A')),
                           '⚠ YES' if d.get('drifted') else '✓ NO'])
    styled_table(drift_rows, col_widths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])

    ev_plots = ev_mon.save_evidently_plots(model_path)
    if 'feature_drift' in ev_plots:
        img(ev_plots['feature_drift'], w=6*inch, h=3*inch)

    # ── Evidently: Classification Performance (Standout #4) ──────────────────
    section("Evidently Monitoring — Classification Performance")
    perf      = ev_results['classification_performance']
    perf_rows = [['Metric', 'Value'],
                 ['Accuracy',  str(perf.get('accuracy', 'N/A'))],
                 ['Precision', str(perf.get('precision', 'N/A'))],
                 ['Recall',    str(perf.get('recall', 'N/A'))],
                 ['F1 Score',  str(perf.get('f1_score', 'N/A'))],
                 ['ROC-AUC',   str(perf.get('roc_auc', 'N/A'))]]
    styled_table(perf_rows, col_widths=[3*inch, 2*inch])
    if 'classification_performance' in ev_plots:
        img(ev_plots['classification_performance'], w=6*inch, h=3*inch)

    # ── Evidently: Prediction Drift (Standout #4) ────────────────────────────
    section("Evidently Monitoring — Prediction / Target Drift")
    pred_dr   = ev_results['prediction_drift']
    story.append(Paragraph(
        f"Prediction drift detected: <b>{pred_dr.get('prediction_drift_detected', '?')}</b>",
        normal))
    story.append(Spacer(1, 0.08*inch))
    ref_dist = pred_dr.get('reference_target_distribution', {})
    cur_dist = pred_dr.get('current_prediction_distribution', {})
    pd_rows  = [['Class', 'Reference Distribution', 'Current Predictions']]
    for cls in sorted(set(list(ref_dist.keys()) + list(cur_dist.keys()))):
        pd_rows.append([f'Class {cls}',
                        f"{ref_dist.get(cls, 0):.4f}",
                        f"{cur_dist.get(cls, 0):.4f}"])
    styled_table(pd_rows, col_widths=[2*inch, 2.5*inch, 2.5*inch])
    if 'prediction_drift' in ev_plots:
        img(ev_plots['prediction_drift'], w=6*inch, h=3*inch)

    # ── Time Trends (Standout #2) ────────────────────────────────────────────
    story.append(PageBreak())
    section("Time Trends — Historical Diagnostics")
    trend_plots = _generate_trend_charts(model_path)
    if trend_plots:
        if 'trend_latency' in trend_plots:
            story.append(Paragraph("Latency over time (ingestion and training):", normal))
            img(trend_plots['trend_latency'], w=6.5*inch, h=3*inch)
        if 'trend_missing' in trend_plots:
            story.append(Paragraph("Average missing value percentage over time:", normal))
            img(trend_plots['trend_missing'], w=6.5*inch, h=3*inch)
    else:
        story.append(Paragraph(
            "No historical snapshots found yet. Trends will appear after multiple fullprocess.py runs.",
            normal))

    # ── Score History from DB (Standout #3 + #2) ─────────────────────────────
    section("Model Score History (from SQL DB)")
    try:
        import dbsetup
        score_hist = dbsetup.get_score_history()
        if score_hist:
            s_rows = [['Recorded At', 'F1 Score', 'Model Path']]
            for sh in score_hist[-10:]:  # last 10
                s_rows.append([sh['scored_at'][:19], f"{sh['f1_score']:.4f}", sh.get('model_path', '')])
            styled_table(s_rows, col_widths=[2.5*inch, 1.5*inch, 3*inch])
        else:
            story.append(Paragraph("No score history in DB yet.", normal))
    except Exception as e:
        story.append(Paragraph(f"DB score history unavailable: {e}", normal))

    doc.build(story)
    print(f"PDF report saved: {pdf_path}")
    return pdf_path


if __name__ == '__main__':
    score_model()
    generate_pdf_report()
