import gradio as gr
import torch
import numpy as np
import nibabel as nib
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.analysis import compute_analysis, longitudinal_comparison
from src.visualization import get_views, get_phase_analysis_plot, get_experiment_progression_plot, get_lesion_wise_tpr_plot

from monai.inferers import sliding_window_inference
from monai.transforms import LoadImaged, EnsureChannelFirstd, ResampleToMatchd, ConcatItemsd, Compose, ScaleIntensityd
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = get_model().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()


def load(file):
    return nib.load(file.name).get_fdata()


monai_transform = Compose([
    LoadImaged(keys=["flair", "t1", "t2"]),
    EnsureChannelFirstd(keys=["flair", "t1", "t2"]),
    ResampleToMatchd(keys=["t1", "t2"], key_dst="flair"),
    ConcatItemsd(keys=["flair", "t1", "t2"], name="image"),
    ScaleIntensityd(keys=["image"])
])


def preprocess_files(flair_path, t1_path, t2_path):
    data = {"flair": flair_path, "t1": t1_path, "t2": t2_path}
    transformed = monai_transform(data)
    return transformed["image"].unsqueeze(0)


def predict(x):
    with torch.no_grad():
        p = sliding_window_inference(x.to(device), (96, 96, 96), 4, model)
        return torch.sigmoid(p).squeeze().cpu().numpy()


def single(flair,t1,t2,thr):
    f_path, t1_path, t2_path = flair.name, t1.name, t2.name
    f = load(flair)
    prob = predict(preprocess_files(f_path, t1_path, t2_path))

    r = compute_analysis(prob,thr)
    fig = get_views(f,r["mask"])

    report_df = pd.DataFrame({
        "Metric": [
            "Total Lesion Volume", 
            "Total Lesion Count", 
            "Small (<10 vox)", 
            "Medium", 
            "Large (>100 vox)"
        ],
        "Value": [
            f"{r['volume']} mm³", 
            str(r['count']), 
            str(r['small']), 
            str(r['medium']), 
            str(r['large'])
        ]
    })

    conf_df = pd.DataFrame({
        "Confidence Metric": [
            "Mean Lesion Confidence", 
            "High Confidence Voxels (>80%)", 
            "Brain Coverage"
        ],
        "Value": [
            f"{r['mean_conf']}%", 
            f"{r['high_conf']}%", 
            f"{r['coverage']}%"
        ]
    })

    return fig, report_df, conf_df


def longitudinal(f1,t11,t21,f2,t12,t22,thr):
    f_path1, t1_path1, t2_path1 = f1.name, t11.name, t21.name
    f_path2, t1_path2, t2_path2 = f2.name, t12.name, t22.name
    
    f1_arr = load(f1)
    f2_arr = load(f2)

    p1 = predict(preprocess_files(f_path1, t1_path1, t2_path1))
    p2 = predict(preprocess_files(f_path2, t1_path2, t2_path2))

    r1,r2,new,res,vol,status = longitudinal_comparison(p1,p2,thr)

    summary_data = [
        ["Total Volume (mm³)", f"{r1['volume']}", f"{r2['volume']}", f"{'+' if vol>0 else ''}{vol}"],
        ["Lesion Count", f"{r1['count']}", f"{r2['count']}", f"{r2['count'] - r1['count']}"],
        ["New / Resolved", "-", "-", f"New: {new} | Res: {res}"],
        ["Clinical Status", "-", "-", status]
    ]
    summary_df = pd.DataFrame(summary_data, columns=["Metric", "Baseline", "Follow-Up", "Change"])

    fig1 = get_views(f1_arr, r1["mask"])
    fig2 = get_views(f2_arr, r2["mask"])

    return fig1, fig2, summary_df


# Apply an attractive and modern Gradio Theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
)

with gr.Blocks(theme=theme, title="MS Lesion AI") as demo:

    gr.Markdown(
        """
        # 🧠 Advanced MS Lesion AI Platform
        ### High-Confidence MRI Segmentation & Longitudinal Tracking
        """
    )

    with gr.Tab("🎯 Single Scan Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input Scans")
                f = gr.File(label="FLAIR Sequence")
                t1 = gr.File(label="T1 Sequence")
                t2 = gr.File(label="T2 Sequence")
                thr = gr.Slider(0.1, 0.9, 0.3, label="Confidence Threshold", info="Adjust model sensitivity")
                
                btn = gr.Button("Analyze Scan 🚀", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 AI Analysis Results")
                gr.Markdown("#### 📸 Multiview Assessment")
                out_img = gr.Plot(show_label=False)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📐 Lesion Morphology Metrics")
                        out_rep = gr.Dataframe(headers=["Metric", "Value"], type="pandas", show_label=False)
                    with gr.Column():
                        gr.Markdown("#### 🎯 Model Confidence")
                        out_conf = gr.Dataframe(headers=["Confidence Metric", "Value"], type="pandas", show_label=False)

        btn.click(single,[f,t1,t2,thr],[out_img,out_rep,out_conf])

    with gr.Tab("📈 Longitudinal Tracking"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🕒 Baseline Study")
                f1 = gr.File(label="FLAIR")
                t11 = gr.File(label="T1")
                t21 = gr.File(label="T2")
                
            with gr.Column():
                gr.Markdown("### 🕒 Follow-Up Study")
                f2 = gr.File(label="FLAIR")
                t12 = gr.File(label="T1")
                t22 = gr.File(label="T2")
                
        thr2 = gr.Slider(0.1, 0.9, 0.3, label="Confidence Threshold")
        btn2 = gr.Button("Compare Studies 🔍", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📸 Baseline Visuals")
                plot1 = gr.Plot(show_label=False)
            with gr.Column():
                gr.Markdown("#### 📸 Follow-Up Visuals")
                plot2 = gr.Plot(show_label=False)

        gr.Markdown("### 📋 Longitudinal Summary Report")
        out_table = gr.Dataframe(headers=["Metric", "Baseline", "Follow-Up", "Change"], type="pandas", show_label=False)

        btn2.click(longitudinal,
                   [f1,t11,t21,f2,t12,t22,thr2],
                   [plot1,plot2,out_table])

    with gr.Tab("📈 Model Performance Metrics"):
        gr.Markdown("### 🌟 Final Model Evaluation Summaries")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📋 Final Test Set Performance")
                perf_df = pd.DataFrame({
                    "Metric": ["Dice Score", "Sensitivity (Recall)", "Specificity", "Hausdorff Distance (95%)", "Volume Error"],
                    "Value": ["0.710", "0.745", "0.995", "5.1 mm", "8.5%"]
                })
                gr.Dataframe(value=perf_df, headers=["Metric", "Value"], type="pandas", show_label=False)
            
            with gr.Column(scale=2):
                gr.Markdown("#### 🚀 Progression Across Experiments")
                gr.Plot(value=get_experiment_progression_plot(), show_label=False)
                
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### ⚖️ Clinical Impact: Baseline vs. Final Model")
                comp_df = pd.DataFrame({
                    "Segmentation Aspect": [
                        "Small Lesion Detection (<10 voxels)", 
                        "False Positives (Artifacts/Noise)", 
                        "Large Lesion Boundaries", 
                        "Longitudinal Tracking"
                    ],
                    "Baseline Model (Dice: 0.63)": [
                        "Misses ~60% of tiny, early-stage lesions", 
                        "Frequent cortex/ventricle misclassifications", 
                        "Under-segments edges by 15-20%", 
                        "Unreliable for tracking 1-2mm changes"
                    ],
                    "Final Model (Dice: 0.71)": [
                        "Captures >75% of early-stage lesions accurately", 
                        "Significantly reduced noise; clean tissue separation", 
                        "Tight conformity to actual lesion borders", 
                        "Reliable measurement for monitoring progression"
                    ]
                })
                gr.Dataframe(value=comp_df, headers=["Segmentation Aspect", "Baseline Model (Dice: 0.63)", "Final Model (Dice: 0.71)"], type="pandas", show_label=False)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("#### 📈 Training Phase Analysis")
                gr.Plot(value=get_phase_analysis_plot(), show_label=False)
            with gr.Column(scale=1):
                gr.Markdown("#### 🎯 Lesion-wise TPR (Detection Profile)")
                gr.Plot(value=get_lesion_wise_tpr_plot(), show_label=False)

demo.launch()