from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "outputs"
OUTPUT = ROOT / "output" / "pdf" / "xai_ids_ieee_paper.pdf"
METADATA_PATH = ROOT / "paper_metadata.yaml"


def paper_metadata() -> dict[str, str]:
    metadata = yaml.safe_load(METADATA_PATH.read_text(encoding="utf-8"))
    required = {"title", "authors", "affiliation", "date", "footer"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Missing paper metadata fields: {sorted(missing)}")
    return {key: str(value) for key, value in metadata.items()}


SERIF = "PaperSerif"
SERIF_BOLD = "PaperSerif-Bold"
SERIF_ITALIC = "PaperSerif-Italic"


def register_fonts() -> None:
    # IEEE Xplore rejects PDFs with non-embedded fonts, so every font used on the
    # page (including table cells and footers) must be a registered TrueType font.
    # The DejaVu fonts ship with matplotlib, which avoids OS-specific font paths.
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    fonts = {
        SERIF: "DejaVuSerif.ttf",
        SERIF_BOLD: "DejaVuSerif-Bold.ttf",
        SERIF_ITALIC: "DejaVuSerif-Italic.ttf",
    }
    for name, filename in fonts.items():
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    # Needed for <b> and <i> markup inside paragraphs to select these faces.
    pdfmetrics.registerFontFamily(
        SERIF, normal=SERIF, bold=SERIF_BOLD, italic=SERIF_ITALIC, boldItalic=SERIF_BOLD
    )


def styles():
    register_fonts()
    serif, bold, italic = SERIF, SERIF_BOLD, SERIF_ITALIC
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=bold,
            fontSize=19,
            leading=21,
            alignment=TA_CENTER,
            spaceAfter=5,
            textColor=colors.black,
        ),
        "author": ParagraphStyle(
            "Author", fontName=serif, fontSize=10, leading=11, alignment=TA_CENTER, spaceAfter=6
        ),
        "abstract": ParagraphStyle(
            "Abstract", fontName=serif, fontSize=8.4, leading=9.6, alignment=TA_JUSTIFY, spaceAfter=4
        ),
        "body": ParagraphStyle(
            "Body",
            fontName=serif,
            fontSize=9.0,
            leading=10.2,
            alignment=TA_JUSTIFY,
            spaceAfter=3.8,
            allowWidows=0,
            allowOrphans=0,
        ),
        "section": ParagraphStyle(
            "Section",
            fontName=bold,
            fontSize=9.6,
            leading=10.8,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            fontName=italic,
            fontSize=9.2,
            leading=10.3,
            alignment=TA_LEFT,
            spaceBefore=3.5,
            spaceAfter=2,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "Caption", fontName=serif, fontSize=7.7, leading=8.6, alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=5
        ),
        "table_caption": ParagraphStyle(
            "TableCaption", fontName=serif, fontSize=7.3, leading=8.2, alignment=TA_CENTER, spaceBefore=4, spaceAfter=2
        ),
        "reference": ParagraphStyle(
            "Reference", fontName=serif, fontSize=7.0, leading=7.55, alignment=TA_LEFT, leftIndent=10, firstLineIndent=-10, spaceAfter=1.0
        ),
        "equation": ParagraphStyle(
            "Equation", fontName=italic, fontSize=8.2, leading=10, alignment=TA_CENTER, spaceBefore=2, spaceAfter=4
        ),
    }


def header_footer(canvas, document):
    metadata = paper_metadata()
    canvas.saveState()
    canvas.setFont(SERIF, 6.5)
    canvas.drawString(0.55 * inch, 0.32 * inch, metadata["footer"])
    canvas.drawRightString(7.95 * inch, 0.32 * inch, str(document.page))
    canvas.restoreState()


def first_page(canvas, document):
    canvas.saveState()
    canvas.setLineWidth(0.4)
    canvas.line(0.55 * inch, 0.43 * inch, 7.95 * inch, 0.43 * inch)
    canvas.setFont(SERIF, 6.5)
    canvas.drawRightString(7.95 * inch, 0.27 * inch, str(document.page))
    canvas.restoreState()


def make_document(path: Path):
    register_fonts()
    metadata = paper_metadata()
    page_w, page_h = letter
    margin = 0.55 * inch
    gutter = 0.22 * inch
    usable = page_w - 2 * margin
    col_w = (usable - gutter) / 2
    # Reserve enough first-page width for the complete title, abstract, and
    # index terms. If this frame is too short, ReportLab flows the abstract
    # into the left body column before the explicit FrameBreak.
    title_bottom = 7.35 * inch
    title_height = page_h - title_bottom - 0.50 * inch
    first_body_top = 7.17 * inch
    body_bottom = 0.52 * inch
    first_body_height = first_body_top - body_bottom
    full_body_height = page_h - body_bottom - 0.54 * inch

    first_frames = [
        Frame(margin, title_bottom, usable, title_height, id="title", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
        Frame(margin, body_bottom, col_w, first_body_height, id="first_left", leftPadding=0, rightPadding=4, topPadding=0, bottomPadding=0),
        Frame(margin + col_w + gutter, body_bottom, col_w, first_body_height, id="first_right", leftPadding=4, rightPadding=0, topPadding=0, bottomPadding=0),
    ]
    later_frames = [
        Frame(margin, body_bottom, col_w, full_body_height, id="left", leftPadding=0, rightPadding=4, topPadding=0, bottomPadding=0),
        Frame(margin + col_w + gutter, body_bottom, col_w, full_body_height, id="right", leftPadding=4, rightPadding=0, topPadding=0, bottomPadding=0),
    ]
    document = BaseDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=metadata["title"],
        author=metadata["authors"],
        subject="Explainable AI, intrusion detection, SHAP, LIME, fidelity, stability, latency",
        keywords="explainable artificial intelligence, network intrusion detection, SHAP, LIME, explanation fidelity",
        # The canvas otherwise starts every page in non-embedded Helvetica.
        initialFontName=SERIF,
    )
    document.addPageTemplates(
        [
            PageTemplate(id="First", frames=first_frames, onPage=first_page, pagesize=letter),
            PageTemplate(id="Later", frames=later_frames, onPage=header_footer, pagesize=letter),
        ]
    )
    return document, col_w


def para(text: str, style, **kwargs):
    return Paragraph(text, style, **kwargs)


def section(number: str, title: str, style):
    return Paragraph(f"{number}. {title.upper()}" if title else number, style)


def subsection(letter: str, title: str, style):
    return Paragraph(f"<i>{letter}. {title}</i>", style)


def make_table(data, widths, font_size=6.2, header=True, alignments=None):
    converted = []
    for row_index, row in enumerate(data):
        converted.append(
            [
                Paragraph(
                    str(value),
                    ParagraphStyle(
                        f"cell_{row_index}_{column_index}",
                        fontName=SERIF_BOLD if header and row_index == 0 else SERIF,
                        fontSize=font_size,
                        leading=font_size + 1,
                        alignment=(alignments[column_index] if alignments else TA_CENTER),
                    ),
                )
                for column_index, value in enumerate(row)
            ]
        )
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    commands = [
        # Tables otherwise select non-embedded Helvetica for every cell.
        ("FONTNAME", (0, 0), (-1, -1), SERIF),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.1),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.65, colors.black),
        ("LINEABOVE", (0, 0), (-1, 0), 0.65, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.65, colors.black),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")))
    table.setStyle(TableStyle(commands))
    return table


def figure(path: Path, width: float, caption: str, number: int, style):
    image = Image(str(path))
    image._restrictSize(width, 3.0 * inch)
    return KeepTogether(
        [image, Paragraph(f"Fig. {number}. {caption}", style)]
    )


def predictive_table(frame: pd.DataFrame, dataset: str, col_w: float, s):
    summary = (
        frame[frame.dataset.eq(dataset)]
        .groupby(["task", "model"])
        .agg(
            accuracy=("accuracy", "mean"),
            balanced=("balanced_accuracy", "mean"),
            macro=("macro_f1", "mean"),
            macro_sd=("macro_f1", "std"),
        )
        .reset_index()
    )
    rows = [["Task", "Model", "Acc.", "Bal. acc.", "Macro-F1"]]
    for row in summary.itertuples():
        rows.append(
            [
                "Bin." if row.task == "binary" else "Multi.",
                "RF" if row.model == "random_forest" else "XGB",
                f"{row.accuracy:.3f}",
                f"{row.balanced:.3f}",
                f"{row.macro:.4f} +/- {row.macro_sd:.4f}",
            ]
        )
    return KeepTogether(
        [
            Paragraph(
                f"TABLE {'III' if dataset == 'CIC-IDS2017' else 'IV'}<br/>{dataset.upper()} PREDICTIVE RESULTS (FIVE SEEDS)",
                s["table_caption"],
            ),
            make_table(
                rows,
                [0.40 * inch, 0.36 * inch, 0.48 * inch, 0.58 * inch, col_w - 1.82 * inch],
                font_size=6.1,
            ),
        ]
    )


def xai_table(xai: pd.DataFrame, paired: pd.DataFrame, dataset: str, col_w: float, s):
    subset = xai[xai.dataset.eq(dataset)]
    rows = [["Task/model", "F-S", "F-L", "Delta [95% CI]", "p-H", "FC S/L", "Med. ms S/L"]]
    for task in ("binary", "multiclass"):
        for model in ("random_forest", "xgboost"):
            pair = paired[
                paired.dataset.eq(dataset) & paired.task.eq(task) & paired.model.eq(model)
            ].iloc[0]
            group = subset[subset.task.eq(task) & subset.model.eq(model)].set_index("explainer")
            shap_row = group.loc["TreeSHAP"]
            lime_row = group.loc["LIME"]
            rows.append(
                [
                    f"{'Bin' if task == 'binary' else 'Multi'}/{'RF' if model == 'random_forest' else 'XGB'}",
                    f"{shap_row.fidelity_mean:.3f}",
                    f"{lime_row.fidelity_mean:.3f}",
                    f"{pair.mean_difference_shap_minus_lime:.3f} [{pair.ci_low:.3f}, {pair.ci_high:.3f}]",
                    "<0.0001" if pair.holm_adjusted_p < 0.0001 else f"{pair.holm_adjusted_p:.4f}",
                    f"{shap_row.faithfulness_correlation_mean:.2f}/{lime_row.faithfulness_correlation_mean:.2f}",
                    f"{shap_row.latency_median_ms:.1f}/{lime_row.latency_median_ms:.1f}",
                ]
            )
    number = "V" if dataset == "CIC-IDS2017" else "VI"
    return KeepTogether(
        [
            Paragraph(
                f"TABLE {number}<br/>{dataset.upper()} EXPLANATION QUALITY AND LATENCY",
                s["table_caption"],
            ),
            make_table(
                rows,
                [0.55 * inch, 0.27 * inch, 0.27 * inch, 0.92 * inch, 0.44 * inch, 0.44 * inch, col_w - 2.89 * inch],
                font_size=5.35,
            ),
        ]
    )


def dataset_protocol_table(col_w: float, s, manifest: dict):
    cic = manifest["datasets"]["CIC-IDS2017/binary"]
    unsw = manifest["datasets"]["UNSW-NB15/binary"]
    cic_test = round(cic["sampled_rows_after_deduplication"] * 0.20)
    cic_train = cic["sampled_rows_after_deduplication"] - cic_test
    rows = [
        ["Dataset", "Protocol", "Rows", "Feat.", "Classes"],
        ["CIC-IDS2017", "Capped, dedup.; strat. 80/20", f"{cic_train:,} / {cic_test:,}", "28", "2 / 15"],
        ["UNSW-NB15", "Predefined; dedup.; overlap-cleaned", f"{unsw['clean_training_rows']:,} / {unsw['clean_testing_rows']:,}", "34", "2 / 10"],
    ]
    return KeepTogether(
        [
            Paragraph("TABLE I<br/>DATASET AND EVALUATION PROTOCOL", s["table_caption"]),
            make_table(
                rows,
                [0.72 * inch, 1.18 * inch, 0.82 * inch, 0.35 * inch, col_w - 3.07 * inch],
                font_size=5.6,
            ),
        ]
    )


def _grid_text(base: dict, grid: dict) -> str:
    values = []
    for key, options in grid.items():
        shown = "/".join("none" if option is None else str(option) for option in options)
        values.append(f"{key.replace('_', ' ')} {shown}")
    return "; ".join(values)


def experiment_settings_table(col_w: float, s, evaluation: dict, config: dict):
    models = config["models"]
    tuning = config["tuning"]
    rows = [
        ["Component", "Setting"],
        [
            "RF grid",
            f"{models['random_forest']['n_estimators']} trees; {_grid_text(models['random_forest'], tuning['random_forest'])}",
        ],
        [
            "XGB grid",
            f"subsample 0.85; {_grid_text(models['xgboost'], tuning['xgboost'])}",
        ],
        [
            "Selection",
            f"Validation macro-F1 on a {tuning['validation_size']:.0%} stratified training split; refit on all training rows",
        ],
        [
            "Explanations",
            f"{evaluation['explanation_samples']} fixed flows x {len(evaluation['explanation_seeds'])} model seeds; "
            f"LIME {evaluation['lime_samples']} samples; k={{{','.join(str(k) for k in evaluation['fidelity_k'])}}}",
        ],
        ["Inference", "2,000 paired bootstrap resamples; Wilcoxon; Holm within each family"],
    ]
    return KeepTogether(
        [
            Paragraph("TABLE II<br/>EXPERIMENT SETTINGS", s["table_caption"]),
            make_table(
                rows,
                [0.72 * inch, col_w - 0.72 * inch],
                font_size=5.8,
                alignments=[TA_LEFT, TA_LEFT],
            ),
        ]
    )


def ablation_table(ablation: pd.DataFrame, col_w: float, s):
    rows = [["Context", "Full k/F1", "Chosen k/F1 (val.)", "Test delta"]]
    for dataset in ("CIC-IDS2017", "UNSW-NB15"):
        for task in ("binary", "multiclass"):
            for model in ("random_forest", "xgboost"):
                group = ablation[
                    ablation.dataset.eq(dataset)
                    & ablation.task.eq(task)
                    & ablation.model.eq(model)
                ]
                full = group.loc[group.feature_count.idxmax()]
                chosen = group[group.selected.eq(1)].iloc[0]
                label = (
                    f"{'CIC' if dataset == 'CIC-IDS2017' else 'UNSW'}/"
                    f"{'Bin' if task == 'binary' else 'Multi'}/"
                    f"{'RF' if model == 'random_forest' else 'XGB'}"
                )
                rows.append(
                    [
                        label,
                        f"{int(full.feature_count)}/{full.test_macro_f1:.4f}",
                        f"{int(chosen.feature_count)}/{chosen.test_macro_f1:.4f}",
                        f"{chosen.test_macro_f1 - full.test_macro_f1:+.4f}",
                    ]
                )
    return KeepTogether(
        [
            Paragraph("TABLE VII<br/>VALIDATION-CHOSEN FEATURE COUNT VERSUS FULL MODEL (TEST MACRO-F1)", s["table_caption"]),
            make_table(
                rows,
                [0.78 * inch, 0.78 * inch, 1.15 * inch, col_w - 2.71 * inch],
                font_size=5.7,
            ),
        ]
    )


def robustness_table(
    baselines: pd.DataFrame,
    balancing: pd.DataFrame,
    calibration: pd.DataFrame,
    masking: pd.DataFrame,
    lime_budget: pd.DataFrame,
    recalibration: pd.DataFrame,
    background: pd.DataFrame,
    sanity: pd.DataFrame,
    conflicts: pd.DataFrame,
    col_w: float,
    s,
):
    baseline_range = baselines[baselines.baseline.eq("shallow_tree_depth5")].macro_f1
    balance = balancing.pivot_table(
        index=["dataset", "task", "model"], columns="training_weighting", values="macro_f1"
    )
    weighted_wins = int((balance["class_weighted"] > balance["none"]).sum())
    ece = calibration.expected_calibration_error
    mask_mean = masking.groupby("baseline").deletion_score.mean()
    lime = lime_budget.groupby("lime_samples").agg(
        stability=("mean_self_stability_cosine", "mean")
    )
    shap_modes = background.groupby("shap_mode").deletion_score.mean()
    sanity_jaccard = sanity.groupby("explainer").top5_jaccard_real_vs_permuted.mean()
    unsw_conflicts = conflicts[
        conflicts.dataset.eq("UNSW-NB15") & conflicts.task.eq("multiclass") & conflicts.partition.eq("test")
    ]
    rows = [
        ["Audit", "Levels", "Main result"],
        ["Simple baseline", "Majority; depth-5 tree", f"Tree macro-F1 {baseline_range.min():.3f}-{baseline_range.max():.3f}"],
        ["Class weighting", "Weighted; none", f"Weighted higher in {weighted_wins}/8 contexts"],
        ["Calibration", "10 binary bins", f"ECE {ece.min():.3f}-{ece.max():.3f}"],
        ["Recalibration", "Isotonic on validation", f"ECE {recalibration.ece_raw.mean():.3f}->{recalibration.ece_isotonic.mean():.3f} (mean)"],
        ["Mask reference", "Median; zero; marginal; neighbors", f"Mean deletion [{mask_mean.min():.3f}, {mask_mean.max():.3f}]"],
        ["LIME budget", "100; 350; 700", f"Stability {lime.stability.iloc[0]:.3f}->{lime.stability.iloc[-1]:.3f}"],
        ["SHAP background", "Path-dep.; interventional", f"Deletion {shap_modes['path_dependent']:.3f} vs {shap_modes['interventional']:.3f} (XGB)"],
        ["Sanity check", "Real vs permuted labels", f"Top-5 overlap S {sanity_jaccard['TreeSHAP']:.2f}; L {sanity_jaccard['LIME']:.2f}"],
        ["Label conflict", "UNSW multiclass test", f"Ceiling {unsw_conflicts.accuracy_ceiling.iloc[0]:.3f}; max class {unsw_conflicts.conflicting_share.max():.2f}"],
    ]
    return KeepTogether(
        [
            Paragraph("TABLE VIII<br/>LITERATURE-DRIVEN ROBUSTNESS AUDITS", s["table_caption"]),
            make_table(rows, [0.25 * col_w, 0.31 * col_w, 0.44 * col_w], font_size=5.7),
        ]
    )


def build_story(col_w: float):
    s = styles()
    metadata = paper_metadata()
    predictive = pd.read_csv(RESULTS / "predictive_metrics.csv")
    per_class = pd.read_csv(RESULTS / "per_class_metrics.csv")
    xai = pd.read_csv(RESULTS / "explanation_summary.csv")
    paired_all = pd.read_csv(RESULTS / "paired_comparisons.csv")
    paired = paired_all[paired_all.metric.eq("deletion")]
    faithfulness_pairs = paired_all[paired_all.metric.eq("faithfulness_correlation")]
    controls_all = pd.read_csv(RESULTS / "fidelity_control_tests.csv")
    controls = controls_all[controls_all.reference.eq("random")]
    global_controls = controls_all[controls_all.reference.eq("global")]
    ablation = pd.read_csv(RESULTS / "feature_ablation.csv")
    tuning = pd.read_csv(RESULTS / "tuning.csv")
    extended = RESULTS / "extended"
    predictive_baselines = pd.read_csv(extended / "predictive_baselines.csv")
    balancing = pd.read_csv(extended / "balancing_ablation.csv")
    calibration = pd.read_csv(extended / "calibration_summary.csv")
    masking = pd.read_csv(extended / "masking_baseline_ablation.csv")
    lime_budget = pd.read_csv(extended / "lime_budget_ablation.csv")
    recalibration = pd.read_csv(extended / "recalibration.csv")
    shap_background = pd.read_csv(extended / "shap_background_ablation.csv")
    sanity = pd.read_csv(extended / "randomization_sanity.csv")
    conflicts = pd.read_csv(extended / "label_conflicts.csv")
    with (RESULTS / "run_manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    evaluation = manifest["config"]["evaluation"]
    predictive_seed_count = len(manifest["config"]["predictive_seeds"])
    predictive_seed_values = ", ".join(str(seed) for seed in manifest["config"]["predictive_seeds"])
    explanation_count = int(evaluation["explanation_samples"])
    lime_repeat_count = int(evaluation.get("lime_repeat_seeds", 3))
    explanation_seed_count = len(evaluation["explanation_seeds"])
    audit_count = int(evaluation["audit_samples"])
    lime_budget_count = int(evaluation["lime_budget_samples"])
    seed_sd_max = float(predictive.groupby(["dataset", "task", "model"]).macro_f1.std().max())
    cic_metadata = manifest["datasets"]["CIC-IDS2017/binary"]

    def passes(frame: pd.DataFrame, explainer: str) -> int:
        selector = frame.explainer.eq(explainer) & (frame.holm_adjusted_p < 0.05) & (frame.ci_low > 0)
        return int(selector.sum())

    global_tree_passes = passes(global_controls, "TreeSHAP")
    global_lime_passes = passes(global_controls, "LIME")
    faith_shap = xai[xai.explainer.eq("TreeSHAP")].faithfulness_correlation_mean
    faith_lime = xai[xai.explainer.eq("LIME")].faithfulness_correlation_mean
    faith_shap_wins = int(
        ((faithfulness_pairs.ci_low > 0) & (faithfulness_pairs.holm_adjusted_p < 0.05)).sum()
    )
    tuned_changed = int(tuning.groupby(["dataset", "task", "model"]).apply(
        lambda g: int(g.loc[g.selected.eq(1), "candidate"].iloc[0] != 0)
    ).sum())
    sanity_summary = sanity.groupby("explainer")[
        ["cosine_real_vs_permuted", "top5_jaccard_real_vs_permuted"]
    ].mean()
    shap_modes = shap_background.groupby("shap_mode").deletion_score.mean()

    unsw_test_conflicts = conflicts[
        conflicts.dataset.eq("UNSW-NB15") & conflicts.task.eq("multiclass") & conflicts.partition.eq("test")
    ].set_index("class")
    unsw_train_conflicts = conflicts[
        conflicts.dataset.eq("UNSW-NB15") & conflicts.task.eq("multiclass") & conflicts.partition.eq("train")
    ].set_index("class")

    pred_summary = predictive.groupby(["dataset", "task", "model"]).mean(numeric_only=True)
    cic_xgb_bin = pred_summary.loc[("CIC-IDS2017", "binary", "xgboost")]
    cic_rf_multi = pred_summary.loc[("CIC-IDS2017", "multiclass", "random_forest")]
    cic_xgb_multi = pred_summary.loc[("CIC-IDS2017", "multiclass", "xgboost")]
    unsw_rf_bin = pred_summary.loc[("UNSW-NB15", "binary", "random_forest")]
    unsw_xgb_bin = pred_summary.loc[("UNSW-NB15", "binary", "xgboost")]
    unsw_rf_multi = pred_summary.loc[("UNSW-NB15", "multiclass", "random_forest")]
    unsw_xgb_multi = pred_summary.loc[("UNSW-NB15", "multiclass", "xgboost")]
    validation_gap = (
        tuning[tuning.selected.eq(1)].set_index(["dataset", "task", "model"]).validation_macro_f1
        - pred_summary.macro_f1
    )
    unsw_bin_gap = float(validation_gap.loc[("UNSW-NB15", "binary")].mean())
    pair_min = paired.mean_difference_shap_minus_lime.min()
    pair_max = paired.mean_difference_shap_minus_lime.max()
    pair_p_max = paired.holm_adjusted_p.max()
    pair_dz_min = paired.paired_effect_dz.min()
    pair_dz_max = paired.paired_effect_dz.max()
    tree_passes = int(
        (controls.explainer.eq("TreeSHAP") & (controls.holm_adjusted_p < 0.05) & (controls.ci_low > 0)).sum()
    )
    lime_passes = int(
        (controls.explainer.eq("LIME") & (controls.holm_adjusted_p < 0.05) & (controls.ci_low > 0)).sum()
    )
    shap_stability = xai[xai.explainer.eq("TreeSHAP")].stability_cosine_mean
    lime_stability = xai[xai.explainer.eq("LIME")].stability_cosine_mean
    xgb_shap_latency = xai[xai.model.eq("xgboost") & xai.explainer.eq("TreeSHAP")].latency_median_ms
    xgb_lime_latency = xai[xai.model.eq("xgboost") & xai.explainer.eq("LIME")].latency_median_ms
    unsw_rf_multi_xai = xai[
        xai.dataset.eq("UNSW-NB15") & xai.task.eq("multiclass") & xai.model.eq("random_forest")
    ].set_index("explainer")
    mask_mean = masking.groupby("baseline").deletion_score.mean()
    lime_budget_summary = lime_budget.groupby("lime_samples").agg(
        fidelity=("mean_deletion_fidelity", "mean"),
        stability=("mean_self_stability_cosine", "mean"),
        latency=("median_latency_ms", "median"),
    )
    balance_comparison = balancing.pivot_table(
        index=["dataset", "task", "model"],
        columns="training_weighting",
        values="macro_f1",
    )
    weighted_wins = int(
        (balance_comparison["class_weighted"] > balance_comparison["none"]).sum()
    )
    failed_lime = controls[
        controls.explainer.eq("LIME")
        & ~((controls.holm_adjusted_p < 0.05) & (controls.ci_low > 0))
    ]
    failed_lime_labels = ", ".join(
        f"{'CIC' if row.dataset == 'CIC-IDS2017' else 'UNSW'} "
        f"{'binary' if row.task == 'binary' else 'multiclass'} "
        f"{'RF' if row.model == 'random_forest' else 'XGBoost'}"
        for row in failed_lime.itertuples()
    )
    unsw_metadata = manifest["datasets"]["UNSW-NB15/binary"]
    ablation_context = []
    reduced_chosen_count = 0
    for _, group in ablation.groupby(["dataset", "task", "model"]):
        full = group.loc[group.feature_count.idxmax()]
        chosen = group[group.selected.eq(1)].iloc[0]
        ablation_context.append(float(chosen.test_macro_f1 - full.test_macro_f1))
        reduced_chosen_count += int(chosen.feature_count < full.feature_count)
    ablation_deltas = np.asarray(ablation_context)

    unsw_seed42_recall = per_class[
        per_class.dataset.eq("UNSW-NB15")
        & per_class.task.eq("multiclass")
        & per_class.seed.eq(42)
    ].pivot(index="class", columns="model", values="recall")
    minority_parts = []
    for attack_class in ("Analysis", "DoS", "Backdoor"):
        if attack_class in unsw_seed42_recall.index:
            row = unsw_seed42_recall.loc[attack_class]
            minority_parts.append(
                f"{attack_class} (RF {row['random_forest']:.3f}; XGBoost {row['xgboost']:.3f})"
            )
    minority_text = ", ".join(minority_parts)

    def context_label(row) -> str:
        return (
            f"{'CIC' if row.dataset == 'CIC-IDS2017' else 'UNSW'} "
            f"{'binary' if row.task == 'binary' else 'multiclass'} "
            f"{'RF' if row.model == 'random_forest' else 'XGBoost'}"
        )

    def failed(frame: pd.DataFrame, explainer: str) -> pd.DataFrame:
        return frame[
            frame.explainer.eq(explainer) & ~((frame.holm_adjusted_p < 0.05) & (frame.ci_low > 0))
        ]

    lime_global_failed = failed(global_controls, "LIME")
    lime_global_failed_labels = "; ".join(context_label(row) for row in lime_global_failed.itertuples())
    lime_global_lift = global_controls[global_controls.explainer.eq("LIME")].mean_lift_top_minus_reference
    shap_global_lift = global_controls[global_controls.explainer.eq("TreeSHAP")].mean_lift_top_minus_reference
    recal_by_dataset = recalibration.groupby("dataset")[["ece_raw", "ece_isotonic"]].mean()
    hard_classes = [name for name in ("Analysis", "Backdoor", "DoS") if name in unsw_test_conflicts.index]
    conflict_text = ", ".join(
        f"{name} {unsw_test_conflicts.loc[name, 'conflicting_share']:.0%}" for name in hard_classes
    )
    unsw_ceiling = float(unsw_test_conflicts.accuracy_ceiling.iloc[0])
    unsw_train_ceiling = float(unsw_train_conflicts.accuracy_ceiling.iloc[0])
    lime_xgb_fidelity = xai[xai.model.eq("xgboost") & xai.explainer.eq("LIME")].set_index(["dataset", "task"]).fidelity_mean
    interventional_by_context = (
        shap_background[shap_background.shap_mode.eq("interventional")]
        .groupby(["dataset", "task"]).deletion_score.mean()
    )
    interventional_beats_lime = int((interventional_by_context > lime_xgb_fidelity.reindex(interventional_by_context.index)).sum())
    sanity_by_context = sanity.groupby(["explainer", "dataset", "task", "model"]).cosine_real_vs_permuted.mean()
    sanity_max_jaccard = float(sanity.groupby(["explainer", "dataset", "task", "model"]).top5_jaccard_real_vs_permuted.mean().max())
    sanity_max_cosine = float(sanity_by_context.max())
    sanity_max_label = context_label(
        pd.Series(dict(zip(["explainer", "dataset", "task", "model"], sanity_by_context.idxmax())))
    )

    attributions = pd.read_csv(RESULTS / "attributions.csv.gz")
    binary_xgb = attributions[
        attributions.task.eq("binary") & attributions.model.eq("xgboost") & attributions.explainer.eq("TreeSHAP")
    ].assign(magnitude=lambda frame: frame.attribution.abs())
    top_features = {
        dataset: ", ".join(group.groupby("feature").magnitude.mean().nlargest(3).index)
        for dataset, group in binary_xgb.groupby("dataset")
    }

    affiliation_line = metadata["affiliation"]
    if metadata["date"]:
        affiliation_line += f" - {metadata['date']}"

    story = [
        Paragraph(
            metadata["title"],
            s["title"],
        ),
        Paragraph(
            f"{metadata['authors']}<br/><font size='8'>{affiliation_line}</font>",
            s["author"],
        ),
        Paragraph(
            f"<b>Abstract-</b> Machine-learning intrusion detection systems can be accurate while providing no defensible account of individual alerts. This work evaluates Random Forest and XGBoost classifiers with TreeSHAP and LIME explanations on CIC-IDS2017 and UNSW-NB15 under a matched, control-based protocol: {explanation_count} common test flows per setting are explained under {explanation_seed_count} model seeds and scored by top-k deletion against random, bottom-ranked, and global-importance controls, by faithfulness correlation, by a label-randomization sanity check, and by stability and latency. Hyperparameters and feature counts are chosen on a training-only validation split. Across {predictive_seed_count} model seeds, CIC binary macro-F1 reached {cic_xgb_bin.macro_f1:.3f} and CIC multiclass {min(cic_rf_multi.macro_f1, cic_xgb_multi.macro_f1):.3f}-{max(cic_rf_multi.macro_f1, cic_xgb_multi.macro_f1):.3f}; on a leakage-controlled UNSW split, binary macro-F1 was {min(unsw_rf_bin.macro_f1, unsw_xgb_bin.macro_f1):.3f}-{max(unsw_rf_bin.macro_f1, unsw_xgb_bin.macro_f1):.3f} and multiclass {min(unsw_rf_multi.macro_f1, unsw_xgb_multi.macro_f1):.3f}-{max(unsw_rf_multi.macro_f1, unsw_xgb_multi.macro_f1):.3f}, where up to {unsw_test_conflicts.conflicting_share.max():.0%} of a class shares exact feature vectors with other labels. TreeSHAP exceeded LIME in all eight paired settings on both deletion fidelity (differences {pair_min:.3f}-{pair_max:.3f}) and faithfulness correlation (Holm-adjusted p <= {faithfulness_pairs.holm_adjusted_p.max():.2g}). TreeSHAP beat random rankings in {tree_passes}/8 and the model's own global ranking in {global_tree_passes}/8 settings; LIME did so in {lime_passes}/8 and only {global_lime_passes}/8, so most LIME rankings add little beyond global importance. Per-alert reports reject unsupported explanations; causal or universal interpretability is not claimed.",
            s["abstract"],
        ),
        Paragraph(
            "<b>Index Terms-</b> Explainable artificial intelligence, network intrusion detection, SHAP, LIME, explanation fidelity.",
            s["abstract"],
        ),
        NextPageTemplate("Later"),
        FrameBreak(),
        section("I", "Introduction", s["section"]),
        para(
            "Modern anomaly-based intrusion detection systems (IDSs) can recognize attack patterns beyond fixed signatures, but their false alarms and opaque decisions complicate security operations [1]. A high-confidence alert is not automatically useful: an analyst must understand which flow characteristics drove the decision, whether the explanation is consistent with similar traffic, and whether it can be produced within an operational latency budget.",
            s["body"],
        ),
        para(
            "SHAP and LIME are widely used post-hoc explainers. SHAP assigns additive feature attributions based on Shapley-value principles [6], whereas LIME fits a sparse surrogate around one prediction [7]. Prior IDS studies commonly present attribution plots or use attributions for feature selection [8]-[10]. Visual plausibility, however, does not establish that highlighted features control model output; explanation research likewise warns that visually convincing attributions can fail basic sanity checks [12], [16]. This work therefore treats explanation quality as a measurable system property.",
            s["body"],
        ),
        para(
            "This study contributes: (1) a leakage-conscious classifier-plus-explainer pipeline spanning two datasets, two task granularities, and two tree model families, with every selection decision made on a training-only validation split; (2) a quantitative XAI protocol that tests top-ranked features against random, bottom-ranked, and global-importance controls, adds faithfulness correlation and a label-randomization sanity check, and repeats every explanation under five model seeds; (3) paired inference with bootstrap intervals, Wilcoxon tests, Holm correction, and standardized effects; (4) a label-conflict audit that separates model failure from unlearnable classes; (5) deployment guidance that jointly considers fidelity, stability, tail latency, minority classes, and feature count; and (6) an operational report that exposes each alert's signed evidence and accepts it only after global and alert-specific random-ranking controls.",
            s["body"],
        ),
        figure(
            RESULTS / "figures" / "paper_workflow.png",
            col_w,
            "End-to-end benchmark workflow. Training-only transformations and fixed held-out partitions precede matched explanation and quantitative validation.",
            1,
            s["caption"],
        ),
        section("II", "Related Work", s["section"]),
        para(
            "Khraisat <i>et al.</i> identify dataset age, false alarms, evasion, and inconsistent evaluation as persistent IDS problems [1]. CIC-IDS2017 was designed to add modern benign activity and diverse attacks across five capture days [2]. UNSW-NB15 contains nine attack families and provides an official 175,341/82,332 training/testing partition [3]. These datasets reduce dependence on legacy KDD-only evaluation, although neither is a direct sample of a production security operations center.",
            s["body"],
        ),
        para(
            "Mane and Rao demonstrated multiple explanation styles for a deep IDS but did not standardize quantitative validation [8]. Gaspar <i>et al.</i> tested whether LIME/SHAP perturbations could change IDS errors [9], while Ahmed <i>et al.</i> combined SHAP feature selection with CIC ensembles [10]. Alazab <i>et al.</i> used information gain to reduce NSL-KDD features [13], and Kasongo and Sun selected 19 UNSW features with XGBoost [14]; neither established a universal feature count. The X-IDS survey by Neupane <i>et al.</i> calls for stakeholder-specific explanations and explicit evaluation metrics [15]. Closest to this work, Arreche <i>et al.</i> compare SHAP and LIME for network IDSs on descriptive accuracy, sparsity, stability, efficiency, robustness, and completeness [17]. Yeh <i>et al.</i> and Adebayo <i>et al.</i> further motivate objective fidelity, sensitivity, and sanity checks beyond visual plausibility [12], [16]. The distinguishing contribution here is not another SHAP/LIME visualization, but matched explainers across model seeds, random and global-importance negative controls, faithfulness correlation [18], a data-randomization sanity check, paired uncertainty, and a reject option for alert-level evidence.",
            s["body"],
        ),
        section("III", "Methodology", s["section"]),
        subsection("A", "Research Questions", s["subsection"]),
        para(
            "RQ1 asks how RF and XGBoost perform across datasets and task granularities. RQ2 asks whether deleting an explainer's top features reduces confidence more than deleting random, bottom-ranked, or globally important features, and whether attribution mass tracks the effect of masking. RQ3 asks whether nearby same-prediction flows receive compatible explanations. RQ4 asks whether latency supports offline, analyst-on-demand, or near-real-time use. RQ5 asks whether conclusions persist across models, datasets, tasks, feature counts, and predictive seeds.",
            s["body"],
        ),
        subsection("B", "Data and Leakage Controls", s["subsection"]),
        para(
            f"The UNSW experiment begins from the predefined 175,341/82,332 training/testing files. Audit found within-partition duplication and exact feature vectors shared across files; exact rows were deduplicated and {unsw_metadata['cross_partition_feature_overlaps_removed_from_test']:,} overlapping test vectors were removed, leaving {unsw_metadata['clean_training_rows']:,}/{unsw_metadata['clean_testing_rows']:,} rows. The accessible CIC representation contains all 2,830,743 flows and 28 selected features but omits day and timestamp metadata. To control computation while retaining minority labels, at most 25,000 rows per label were sampled, {cic_metadata['duplicates_removed_after_sampling']:,} sampled duplicates were removed, and the remaining {cic_metadata['sampled_rows_after_deduplication']:,} rows were split 80/20 with label stratification. Neither dataset result is claimed as temporal generalization evidence.",
            s["body"],
        ),
        para(
            "All preprocessing is learned from training data. Infinite numeric values are converted to missing values and replaced with training medians. Categorical variables are ordinal-encoded with training-derived dictionaries and an explicit unseen category. Labels and attack categories are excluded from the feature space. Binary mappings use Normal versus Attack; multiclass experiments retain the supplied labels. After duplicate/overlap removal, the naturally imbalanced UNSW test distribution receives no class resampling or balancing.",
            s["body"],
        ),
        dataset_protocol_table(col_w, s, manifest),
        Spacer(1, 4),
        subsection("C", "Models and Predictive Measures", s["subsection"]),
        para(
            f"RF uses balanced bootstrap class weights [4]; XGBoost uses histogram training, 0.85 row/column subsampling, and balanced sample weights [5]. For each dataset, task, and model, eight candidate settings (Table II), the prespecified one first, are fitted on {1 - manifest['config']['tuning']['validation_size']:.0%} of the training rows and compared by macro-F1 on the remaining stratified {manifest['config']['tuning']['validation_size']:.0%}; ties keep the earlier candidate. The selected setting, which differed from the prespecified one in {tuned_changed}/8 contexts, is refitted on all training rows with seeds {predictive_seed_values}. The test partition is never used for selection. Accuracy, balanced accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrices, and binary PR-AUC, false-positive rate (FPR), and false-negative rate (FNR) are reported.",
            s["body"],
        ),
        experiment_settings_table(col_w, s, evaluation, manifest["config"]),
        Spacer(1, 4),
        subsection("D", "Explanation Metrics", s["subsection"]),
        para(
            f"For each configuration, the same {explanation_count} fixed, label-stratified test flows, chosen once from the seed-42 model, are explained by both methods under each of the {explanation_seed_count} model seeds; available binary true-positive, false-positive, true-negative, and false-negative cases are explicitly represented. Attributions are converted to signed vectors in the original feature order. For an input <i>x</i>, predicted class <i>c</i>, feature ranking <i>r</i>, and training baseline <i>b</i>, the deletion score at <i>k</i> is",
            s["body"],
        ),
        para(
            "D<sub>k</sub>(x,r) = p(c|x) - p(c|mask(x, r<sub>1:k</sub>, b)).",
            s["equation"],
        ),
        para(
            f"The primary fidelity score averages D<sub>k</sub> for k in {{1,3,5,10}}. The same calculation uses three controls: a fixed random ranking, the bottom attribution ranking, and a global ranking given by the model's own feature importances, identical for every flow. A local explanation that cannot beat the global ranking adds no per-alert information. Flip rate records whether masking changes the predicted class. Because deletion only probes the head of the ranking, faithfulness correlation [18] is also computed: {evaluation['faithfulness_subsets']} random subsets of {evaluation['faithfulness_subset_size']} features are masked per flow, and the Pearson correlation between each subset's summed attribution and its probability drop is recorded. Per-flow scores are averaged over model seeds before 2,000 paired bootstrap resamples over flows.",
            s["body"],
        ),
        para(
            f"Within each {explanation_count}-flow evaluation subset, neighborhood stability pairs each explained flow with its nearest standardized flow having the same predicted class, then reports signed cosine similarity and top-5 Jaccard overlap. LIME self-stability is estimated from {lime_repeat_count} seeds on {evaluation['self_stability_samples']} repeated flows. Explainer-call latency is measured per instance after one warm-up and summarized by median, P90, P95, P99, and throughput; it excludes data loading, preprocessing, prediction, serialization, and report rendering. LIME uses {evaluation['lime_samples']} perturbation samples; TreeSHAP uses the fitted tree model directly.",
            s["body"],
        ),
        para(
            "For explanation vectors a and a' assigned to nearest same-prediction flows, cosine stability is S<sub>cos</sub>=(a dot a')/(||a|| ||a'||); rank stability is the Jaccard overlap between their top-5 feature sets. These measures deliberately separate attribution magnitude/direction from rank agreement. A stable explanation is not necessarily faithful, so neither measure substitutes for deletion controls.",
            s["body"],
        ),
        subsection("E", "Statistical Analysis and Decision Rules", s["subsection"]),
        para(
            "Because TreeSHAP and LIME explain identical flows, the primary comparison uses within-flow differences in mean top-k deletion score. The mean difference and a percentile 95% confidence interval are obtained with 2,000 paired bootstrap resamples. The prespecified directional hypothesis (TreeSHAP fidelity greater than LIME) is tested with a one-sided Wilcoxon signed-rank test. Holm's step-down procedure controls family-wise error within each family: eight deletion comparisons, eight two-sided faithfulness-correlation comparisons (a metric added after the original protocol), 16 top-versus-random tests, and 16 top-versus-global tests. Statistical significance requires adjusted p < 0.05; practical importance is reported as the raw probability-drop difference and paired d<sub>z</sub>, defined as the mean paired difference divided by its sample standard deviation. These quantities describe the fixed evaluation sample and intervention, not a population of future networks.",
            s["body"],
        ),
        figure(
            RESULTS / "figures" / "predictive_performance.png",
            col_w,
            f"Macro-F1 by dataset, task, and model. Bars aggregate {predictive_seed_count} predictive seeds.",
            2,
            s["caption"],
        ),
        section("IV", "Experimental Results", s["section"]),
        subsection("A", "Predictive Performance", s["subsection"]),
        predictive_table(predictive, "CIC-IDS2017", col_w, s),
        Spacer(1, 4),
        para(
            f"CIC binary detection is strong: XGBoost achieves {cic_xgb_bin.accuracy:.4f} accuracy and {cic_xgb_bin.macro_f1:.4f} macro-F1 with FPR {cic_xgb_bin.false_positive_rate:.4f} and FNR {cic_xgb_bin.false_negative_rate:.4f}. Multiclass performance remains high but lower, with macro-F1 {cic_rf_multi.macro_f1:.4f} for RF and {cic_xgb_multi.macro_f1:.4f} for XGBoost. Seed standard deviations are at most {seed_sd_max:.4f} macro-F1, indicating low training randomness relative to dataset effects.",
            s["body"],
        ),
        predictive_table(predictive, "UNSW-NB15", col_w, s),
        Spacer(1, 4),
        para(
            f"The leakage-controlled UNSW partition remains substantially harder. Binary macro-F1 is {unsw_rf_bin.macro_f1:.4f} (RF) and {unsw_xgb_bin.macro_f1:.4f} (XGBoost). XGBoost has FNR {unsw_xgb_bin.false_negative_rate:.4f} and FPR {unsw_xgb_bin.false_positive_rate:.4f}. Multiclass macro-F1 is {unsw_rf_multi.macro_f1:.4f} (RF) and {unsw_xgb_multi.macro_f1:.4f} (XGBoost). The macro/weighted gap shows that aggregate performance is influenced by frequent classes.",
            s["body"],
        ),
        subsection("B", "Explanation Fidelity and Stability", s["subsection"]),
        xai_table(xai, paired, "CIC-IDS2017", col_w, s),
        Spacer(1, 4),
        xai_table(xai, paired, "UNSW-NB15", col_w, s),
        Spacer(1, 4),
        para(
            f"TreeSHAP has higher median/mode-baseline top-k deletion fidelity than LIME in every configuration. Paired mean improvements range from {pair_min:.3f} to {pair_max:.3f}; all eight bootstrap intervals exclude zero, paired d<sub>z</sub> ranges from {pair_dz_min:.3f} to {pair_dz_max:.3f}, and every Holm-adjusted one-sided p-value is at most {pair_p_max:.2g}. Faithfulness correlation agrees across the whole ranking: TreeSHAP averages {faith_shap.min():.2f}-{faith_shap.max():.2f} against {faith_lime.min():.2f}-{faith_lime.max():.2f} for LIME, and the paired difference is significant in {faith_shap_wins}/8 configurations.",
            s["body"],
        ),
        para(
            f"The control analysis is stricter. Against random rankings, TreeSHAP passes in {tree_passes}/8 configurations after Holm correction and LIME in {lime_passes}/8 (failing for {failed_lime_labels or 'none'}). The global-importance control separates the explainers more sharply: TreeSHAP's per-flow rankings beat the model's own global ranking in {global_tree_passes}/8 configurations (mean lift {shap_global_lift.min():.3f}-{shap_global_lift.max():.3f}), whereas LIME does so in only {global_lime_passes}/8, with lifts of {lime_global_lift.min():+.3f} to {lime_global_lift.max():+.3f}. In {lime_global_failed_labels}, an analyst would learn as much from one fixed importance list as from LIME's per-alert explanation.",
            s["body"],
        ),
        para(
            f"TreeSHAP's neighborhood cosine similarity ranges from {shap_stability.min():.3f} to {shap_stability.max():.3f}, compared with {lime_stability.min():.3f} to {lime_stability.max():.3f} for LIME. The findings reject the idea that a single attractive local bar chart is sufficient validation: an explainer can look plausible while failing a negative control or changing materially across nearby flows.",
            s["body"],
        ),
        figure(
            RESULTS / "figures" / "paper_paired_fidelity.png",
            col_w,
            "Paired TreeSHAP-minus-LIME differences in deletion fidelity (top) and faithfulness correlation (bottom) over shared flows, averaged across model seeds, with 95% bootstrap intervals. Deletion p-values in Tables V-VI are Holm-adjusted Wilcoxon results; FC is mean faithfulness correlation.",
            3,
            s["caption"],
        ),
        subsection("C", "Latency and Operational Fit", s["subsection"]),
        para(
            f"Latency depends on the model-explainer pair. For XGBoost, TreeSHAP median latency is {xgb_shap_latency.min():.1f}-{xgb_shap_latency.max():.1f} ms, compared with {xgb_lime_latency.min():.1f}-{xgb_lime_latency.max():.1f} ms for LIME. For UNSW RF multiclass, medians are {unsw_rf_multi_xai.loc['TreeSHAP'].latency_median_ms:.1f} ms for TreeSHAP and {unsw_rf_multi_xai.loc['LIME'].latency_median_ms:.1f} ms for LIME; TreeSHAP P95 is {unsw_rf_multi_xai.loc['TreeSHAP'].latency_p95_ms:.1f} ms. Deployment guidance therefore cannot be based on explainer name alone.",
            s["body"],
        ),
        figure(
            RESULTS / "figures" / "paper_latency_configurations.png",
            col_w,
            "Median single-flow latency for every model-explainer configuration on the fixed host. The horizontal axis is logarithmic.",
            4,
            s["caption"],
        ),
        subsection("D", "Feature-Count Ablation and Error Analysis", s["subsection"]),
        para(
            f"The feature count is chosen on the validation split as the smallest tested count whose validation macro-F1 is within 0.005 of the full model; the test partition is used only to report the chosen count. A reduced set is chosen in {reduced_chosen_count}/8 contexts, and chosen-minus-full test macro-F1 ranges from {ablation_deltas.min():+.3f} to {ablation_deltas.max():+.3f}. The useful count remains context-specific, so no universally optimal k is implied.",
            s["body"],
        ),
        ablation_table(ablation, col_w, s),
        Spacer(1, 4),
        figure(
            RESULTS / "figures" / "feature_ablation.png",
            col_w,
            "Feature-count ablation on test data. Rankings come from a model fitted on the training fit split; each subset is retrained with seed 42, and the reported count is chosen on validation data.",
            5,
            s["caption"],
        ),
        para(
            f"UNSW minority failures remain operationally important. Seed-42 recall for selected difficult classes is {minority_text}. The label-conflict audit shows that much of this error is unlearnable from the supplied features: in the test partition, {conflict_text} of rows share an exact feature vector with a differently labelled row, which caps achievable multiclass accuracy at {unsw_ceiling:.3f} on test and {unsw_train_ceiling:.3f} on training data. These classes should be reported as dataset limitations rather than model deficits. CIC errors concentrate in Web Attack classes, but supports are small, so those rates have high sampling uncertainty. Global attributions are dataset-specific: averaged over the explained flows, binary XGBoost TreeSHAP ranks {top_features['CIC-IDS2017']} highest on CIC, but {top_features['UNSW-NB15']} on UNSW.",
            s["body"],
        ),
        figure(
            RESULTS / "figures" / "paper_unsw_recall.png",
            col_w,
            "UNSW-NB15 multiclass recall at seed 42 beside each class's label-conflict share (test rows whose exact feature vector also carries another label). The weakest classes are those with the most conflicting rows.",
            6,
            s["caption"],
        ),
        subsection("E", "Literature-Driven Robustness Audits", s["subsection"]),
        robustness_table(
            predictive_baselines, balancing, calibration, masking, lime_budget,
            recalibration, shap_background, sanity, conflicts, col_w, s,
        ),
        para(
            f"Both full model families exceed the majority and depth-5 tree baselines. With identical seed 42 fits, class weighting improves macro-F1 in {weighted_wins}/8 contexts, so balancing is not universally beneficial. Seed-42 binary expected calibration error is {calibration[calibration.dataset.eq('CIC-IDS2017')].expected_calibration_error.min():.3f}-{calibration[calibration.dataset.eq('CIC-IDS2017')].expected_calibration_error.max():.3f} on CIC and {calibration[calibration.dataset.eq('UNSW-NB15')].expected_calibration_error.min():.3f}-{calibration[calibration.dataset.eq('UNSW-NB15')].expected_calibration_error.max():.3f} on UNSW. On {audit_count} flows per configuration, TreeSHAP deletion remains large with median/mode ({mask_mean['training_median_mode']:.3f}), zero ({mask_mean['zero']:.3f}), marginal-row ({mask_mean['marginal_training_row']:.3f}), and nearest-opposite ({mask_mean['nearest_opposite_prediction']:.3f}) references, but collapses for nearest same-prediction replacement ({mask_mean['nearest_same_prediction']:.3f}); fidelity is therefore an intervention-specific estimand. On {lime_budget_count} flows per configuration and {lime_repeat_count} repeats, increasing LIME from 100 to 700 samples raises pooled self-stability from {lime_budget_summary.loc[100].stability:.3f} to {lime_budget_summary.loc[700].stability:.3f}, while median latency rises from {lime_budget_summary.loc[100].latency:.1f} to {lime_budget_summary.loc[700].latency:.1f} ms and mean deletion fidelity from {lime_budget_summary.loc[100].fidelity:.3f} to {lime_budget_summary.loc[700].fidelity:.3f}, still below TreeSHAP. Isotonic recalibration fitted on validation data lowers mean ECE from {recal_by_dataset.loc['CIC-IDS2017', 'ece_raw']:.3f} to {recal_by_dataset.loc['CIC-IDS2017', 'ece_isotonic']:.3f} on CIC but only from {recal_by_dataset.loc['UNSW-NB15', 'ece_raw']:.3f} to {recal_by_dataset.loc['UNSW-NB15', 'ece_isotonic']:.3f} on UNSW, where validation macro-F1 exceeds binary test macro-F1 by {unsw_bin_gap:.3f}: the predefined UNSW partitions differ in distribution, so calibration learned on training data transfers poorly. For XGBoost, interventional TreeSHAP with a 100-row background [19] gives mean deletion {shap_modes['interventional']:.3f} versus {shap_modes['path_dependent']:.3f} for the default path-dependent mode, with the ordering varying by dataset; interventional TreeSHAP still exceeds LIME's mean deletion in {interventional_beats_lime}/{len(interventional_by_context)} XGBoost configurations, so the explainer comparison does not hinge on the background choice. In the label-randomization sanity check [16], explanations of the real model and of a model trained on permuted labels have a mean top-5 Jaccard overlap of {sanity_summary.loc['TreeSHAP', 'top5_jaccard_real_vs_permuted']:.2f} (TreeSHAP) and {sanity_summary.loc['LIME', 'top5_jaccard_real_vs_permuted']:.2f} (LIME), at most {sanity_max_jaccard:.2f} in any configuration, so both explainers respond to what the model learned rather than to input structure alone. Signed cosine is also low except for {sanity_max_label} ({sanity_max_cosine:.2f}), where both models weight the same high-variance features.",
            s["body"],
        ),
        section("V", "Discussion", s["section"]),
        subsection("A", "Answers to the Research Questions", s["subsection"]),
        para(
            f"RQ1: validation-tuned RF and XGBoost beat simple baselines and are nearly tied, but UNSW multiclass detection is bounded by label conflicts in the data. RQ2: TreeSHAP clears the random control in {tree_passes}/8 and the global-importance control in {global_tree_passes}/8 configurations and exceeds LIME on deletion and faithfulness correlation everywhere; LIME clears the global control in only {global_lime_passes}/8. Absolute deletion still depends on the reference intervention. RQ3: TreeSHAP is more neighbor-consistent within the selected subsets; LIME stability improves with sampling budget. RQ4: no explainer has a universal latency advantage. RQ5: the paired TreeSHAP advantage persists across datasets, tasks, models, and five model seeds, while calibration, balancing, feature count, masking magnitude, and latency remain context-dependent.",
            s["body"],
        ),
        subsection("B", "Why the Explainers Differ", s["subsection"]),
        para(
            "TreeSHAP exploits the fitted tree structure and allocates credit consistently under its conditional-expectation assumptions. LIME instead samples a local neighborhood, discretizes features, and fits a surrogate. In mixed continuous/categorical flow spaces, that synthetic neighborhood may not match the data manifold; limited sampling also introduces variance. These mechanisms plausibly explain TreeSHAP's stronger fidelity and stability, and why LIME's rankings often collapse toward the features the model uses everywhere, which the global control then cannot distinguish from a fixed importance list. The experiment does not isolate which LIME component is responsible. Attribution magnitudes are not compared directly across methods because SHAP values and surrogate coefficients have different scales; only rankings and downstream behavioral tests are compared.",
            s["body"],
        ),
        subsection("C", "Operational Meaning", s["subsection"]),
        para(
            "The benchmark separates three questions that are often conflated. Predictive quality asks whether the IDS labels traffic correctly. Fidelity asks whether an explanation reflects the fitted model. Human usefulness asks whether an analyst can make a better decision. This work measures the first two and operational cost; it does not claim that a faithful explanation is automatically understandable, causal, or actionable. That distinction is central when using XAI in a security operations workflow.",
            s["body"],
        ),
        section("VI", "Deployment Recommendations", s["section"]),
        para(
            f"Classification should run on every flow, while explanation generation should be policy-triggered. XGBoost plus TreeSHAP is the strongest measured low-latency candidate because it combines high primary fidelity with {xgb_shap_latency.min():.1f}-{xgb_shap_latency.max():.1f} ms median explainer-call time. CIC RF plus TreeSHAP is suitable for analyst-on-demand use. UNSW RF TreeSHAP should be reserved for lower-volume investigation unless its {unsw_rf_multi_xai.loc['TreeSHAP'].latency_p95_ms:.1f} ms multiclass tail is acceptable. LIME can be faster than RF TreeSHAP, but its rankings fail the random control in {len(failed_lime)} and the global-importance control in {len(lime_global_failed)} of 8 configurations; speed alone does not justify deployment.",
            s["body"],
        ),
        para(
            "The supplied transparency layer produces Markdown and JSON records containing predicted class and probability, top signed features and observed values, the training-derived masking baseline, local top/bottom/random deletion evidence, an empirical p-value, and benchmark stability and latency context. It labels evidence supported only when TreeSHAP passed the corrected global control and the alert clears a local randomization test; otherwise it requires review. This operationally mitigates decision opacity while explicitly warning that attribution is not causality. Governance should also monitor drift and class-specific fidelity.",
            s["body"],
        ),
        section("VII", "Threats to Validity", s["section"]),
        subsection("A", "Data and External Validity", s["subsection"]),
        para(
            "The CIC mirror contains a 28-feature preprocessing and no day metadata; therefore, its stratified split can still benefit from capture-specific artifacts and should not be interpreted as temporal generalization. Sampling was performed before deduplication to control memory, although duplicates were removed before splitting. Engelen <i>et al.</i> reconstructed or relabelled more than 20% of CIC-IDS2017 flows after finding flow-construction and labelling errors [20]; the accessible mirror predates that correction, so CIC scores may partly reflect those artifacts. UNSW uses a predefined partition derived from one testbed rather than a documented temporal holdout. Duplicate and cross-partition overlap removal reduces direct leakage but cannot make either benchmark representative of a live network.",
            s["body"],
        ),
        subsection("B", "Construct Validity", s["subsection"]),
        para(
            f"Masking can create implausible combinations because flow variables are correlated. The reference audit makes this dependence explicit: pooled top-k deletion is {mask_mean['training_median_mode']:.3f} for median/mode and {mask_mean['nearest_opposite_prediction']:.3f} for a nearest opposite prediction, but only {mask_mean['nearest_same_prediction']:.3f} for a nearest same prediction. These interventions answer different questions, and none proves causality. Cosine similarity may also penalize appropriate boundary changes. Most importantly, the study measures fitted-model faithfulness rather than human comprehension or improved analyst decisions.",
            s["body"],
        ),
        subsection("C", "Statistical Validity", s["subsection"]),
        para(
            f"Explanation inference uses {explanation_count} matched flows per configuration, each explained under {explanation_seed_count} model seeds. Paired analysis and multiplicity correction improve efficiency and false-positive control, but intervals describe the selected flows rather than repeated datasets. Flow-level resampling also treats selected flows as independent even though shared sessions or capture conditions may induce dependence, so intervals may be optimistic. The neighbor metric searches only within those selected flows and uses standardized Euclidean distance on a mixed encoded feature space. Rare classes have small support. The {predictive_seed_count} predictive seeds quantify algorithmic variation, not uncertainty caused by alternative captures, preprocessing pipelines, or hyperparameter searches.",
            s["body"],
        ),
        subsection("D", "Scope", s["subsection"]),
        para(
            "Only tree classifiers and two post-hoc explainers are evaluated. The LIME budget audit covers 100-700 samples but not alternative kernels or discretizers. Recalibration is evaluated but not applied to the alert scores. SMOTE is omitted because interpolating encoded categorical flow fields can create invalid traffic; weighting versus no weighting is reported instead. Deep models, counterfactuals, calibrated thresholds, drift, adversarial manipulation, causal claims, and analyst studies remain outside scope.",
            s["body"],
        ),
        subsection("E", "Ethics and Artifact Availability", s["subsection"]),
        para(
            "The study uses public benchmark flow records generated in controlled testbeds and performs no human-participant or animal research, live-network intervention, or attack deployment. Dataset redistribution remains subject to the providers' terms. To preserve double-blind review, the manuscript omits an identifying repository link; the camera-ready version should provide an archival code URL and DOI together with the exact commit. The released artifact excludes raw data and serialized models but includes acquisition guidance, source hashes, cleaning audits, configurations, tests, and regenerated results.",
            s["body"],
        ),
        section("VIII", "Reproducibility", s["section"]),
        para(
            f"The run used Python {manifest['environment']['python']}, scikit-learn {manifest['environment']['scikit_learn']}, XGBoost {manifest['environment']['xgboost']}, and SHAP {manifest['environment']['shap']} on {manifest['environment']['platform'].split('-')[0]} with {manifest['environment']['logical_cpu_count']} logical CPUs. The artifact stores the fixed configuration, package pins, SHA-256 source hashes, row and deduplication audits, per-instance metrics, comparisons, plots, unit tests, cross-artifact checks, and a machine-readable manifest. Hyperparameters come from the recorded validation-split search (tuning.csv), and the selected settings are stored in the manifest. Scripts regenerate excluded raw-data and serialized-model artifacts from locally supplied inputs.",
            s["body"],
        ),
        section("IX", "Conclusion", s["section"]),
        para(
            f"This study addresses black-box IDS decisions within a defined operational scope by evaluating detection and explanation as one system and producing rejectable per-alert evidence. Under a protocol with validation-only selection, 200 matched flows, and five model seeds, TreeSHAP exceeds LIME in all eight paired comparisons on both deletion fidelity and faithfulness correlation, clears {tree_passes}/8 random and {global_tree_passes}/8 global-importance controls, and has higher neighborhood consistency; LIME clears {lime_passes}/8 and {global_lime_passes}/8. Both explainers pass a label-randomization sanity check. The weakest UNSW classes are largely unlearnable from the supplied features, which the label-conflict audit makes explicit. The transparency report converts this evidence into auditable alerts, but reference sensitivity confirms that post-hoc attribution is not causal or universal interpretability. Future work should add realistic conditional interventions, corrected CIC-IDS2017 flows [20], capture-aware resampling, drift tests, more explainers, and an analyst study.",
            s["body"],
        ),
        section("REFERENCES", "", s["section"]),
    ]

    references = [
        "[1] A. Khraisat, I. Gondal, P. Vamplew, and J. Kamruzzaman, \"Survey of intrusion detection systems: Techniques, datasets and challenges,\" <i>Cybersecurity</i>, vol. 2, art. no. 20, 2019, doi: 10.1186/s42400-019-0038-7.",
        "[2] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, \"Toward generating a new intrusion detection dataset and intrusion traffic characterization,\" in <i>Proc. 4th Int. Conf. Information Systems Security and Privacy (ICISSP)</i>, 2018, pp. 108-116, doi: 10.5220/0006639801080116.",
        "[3] N. Moustafa and J. Slay, \"UNSW-NB15: A comprehensive data set for network intrusion detection systems,\" in <i>Proc. Military Communications and Information Systems Conf. (MilCIS)</i>, 2015, pp. 1-6, doi: 10.1109/MilCIS.2015.7348942.",
        "[4] L. Breiman, \"Random forests,\" <i>Machine Learning</i>, vol. 45, pp. 5-32, 2001, doi: 10.1023/A:1010933404324.",
        "[5] T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 785-794, doi: 10.1145/2939672.2939785.",
        "[6] S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in <i>Advances in Neural Information Processing Systems</i>, vol. 30, 2017, pp. 4765-4774.",
        "[7] M. T. Ribeiro, S. Singh, and C. Guestrin, \"Why should I trust you?: Explaining the predictions of any classifier,\" in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 1135-1144, doi: 10.1145/2939672.2939778.",
        "[8] S. Mane and D. Rao, \"Explaining network intrusion detection system using explainable AI framework,\" <i>CoRR</i>, abs/2103.07110, 2021.",
        "[9] D. Gaspar, P. Silva, and C. Silva, \"Explainable AI for intrusion detection systems: LIME and SHAP applicability on multi-layer perceptron,\" <i>IEEE Access</i>, vol. 12, pp. 30164-30175, 2024, doi: 10.1109/ACCESS.2024.3368377.",
        "[10] U. Ahmed <i>et al.</i>, \"Hybrid bagging and boosting with SHAP based feature selection for enhanced predictive modeling in intrusion detection systems,\" <i>Scientific Reports</i>, vol. 14, art. no. 30532, 2024, doi: 10.1038/s41598-024-81151-1.",
        "[11] F. Pedregosa <i>et al.</i>, \"Scikit-learn: Machine learning in Python,\" <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825-2830, 2011.",
        "[12] C.-K. Yeh, C.-Y. Hsieh, A. Suggala, D. I. Inouye, and P. Ravikumar, \"On the (in)fidelity and sensitivity of explanations,\" in <i>Advances in Neural Information Processing Systems</i>, vol. 32, 2019.",
        "[13] A. Alazab, M. Hobbs, J. Abawajy, and M. Alazab, \"Using feature selection for intrusion detection system,\" in <i>Proc. Int. Symp. Communications and Information Technologies</i>, 2012, pp. 296-301, doi: 10.1109/ISCIT.2012.6380910.",
        "[14] S. M. Kasongo and Y. Sun, \"Performance analysis of intrusion detection systems using a feature selection method on the UNSW-NB15 dataset,\" <i>Journal of Big Data</i>, vol. 7, art. no. 105, 2020, doi: 10.1186/s40537-020-00379-6.",
        "[15] S. Neupane <i>et al.</i>, \"Explainable intrusion detection systems (X-IDS): A survey of current methods, challenges, and opportunities,\" <i>IEEE Access</i>, vol. 10, pp. 112392-112415, 2022, doi: 10.1109/ACCESS.2022.3216617.",
        "[16] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, \"Sanity checks for saliency maps,\" in <i>Advances in Neural Information Processing Systems</i>, vol. 31, 2018.",
        "[17] O. Arreche, T. R. Guntur, J. W. Roberts, and M. Abdallah, \"E-XAI: Evaluating black-box explainable AI frameworks for network intrusion detection,\" <i>IEEE Access</i>, vol. 12, pp. 23954-23988, 2024, doi: 10.1109/ACCESS.2024.3365140.",
        "[18] U. Bhatt, A. Weller, and J. M. F. Moura, \"Evaluating and aggregating feature-based model explanations,\" in <i>Proc. 29th Int. Joint Conf. Artificial Intelligence (IJCAI)</i>, 2020, pp. 3016-3022, doi: 10.24963/ijcai.2020/417.",
        "[19] H. Chen, J. D. Janizek, S. Lundberg, and S.-I. Lee, \"True to the model or true to the data?\" arXiv:2006.16234, 2020.",
        "[20] G. Engelen, V. Rimmer, and W. Joosen, \"Troubleshooting an intrusion detection dataset: The CICIDS2017 case study,\" in <i>Proc. IEEE Security and Privacy Workshops (SPW)</i>, 2021, pp. 7-12, doi: 10.1109/SPW53761.2021.00009.",
    ]
    story.extend(Paragraph(reference, s["reference"]) for reference in references)
    return story


def main() -> None:
    from src.xai_ids.benchmark import make_summary_plots
    from src.xai_ids.paper_figures import build_all

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Summary plots read only the committed CSVs, so the paper can be rebuilt
    # without re-running the full benchmark.
    make_summary_plots(RESULTS)
    build_all(RESULTS)
    document, col_w = make_document(OUTPUT)
    document.build(build_story(col_w))
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
