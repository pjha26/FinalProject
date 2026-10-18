"""Build the implementation PDF from saved local evidence using ReportLab.

Run with a documentation Python containing reportlab, Pillow and pypdf.
These packages are not additional inference-runtime dependencies.
"""
import argparse
import json
from pathlib import Path
import subprocess
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Preformatted, Flowable

ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, GREY = colors.HexColor('#17324d'), colors.HexColor('#087e8b'), colors.HexColor('#52616f')
WIDTH = 174 * mm
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle', fontName='Helvetica-Bold', fontSize=25, leading=30, textColor=NAVY, spaceAfter=14))
styles.add(ParagraphStyle(name='ReportHeading', fontName='Helvetica-Bold', fontSize=16, leading=21, textColor=NAVY, spaceAfter=12))
styles.add(ParagraphStyle(name='ReportBody', fontName='Helvetica', fontSize=10, leading=15, spaceAfter=9, textColor=NAVY))
styles.add(ParagraphStyle(name='ReportSmall', fontName='Helvetica', fontSize=8.5, leading=12, textColor=GREY, spaceAfter=7))
styles.add(ParagraphStyle(name='ReportCell', fontName='Helvetica', fontSize=9, leading=12, textColor=NAVY))
styles.add(ParagraphStyle(name='ReportCode', fontName='Courier', fontSize=8, leading=12, textColor=NAVY))


def p(text, style='ReportBody'):
    return Paragraph(text, styles[style])


def table(rows, widths):
    values = [[p(escape(str(cell)), 'ReportCell') for cell in row] for row in rows]
    result = Table(values, colWidths=widths, repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dceef0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f6f8')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, 0), .8, TEAL),
    ]))
    return result


class Architecture(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = WIDTH, 105 * mm

    def draw(self):
        c = self.canv
        boxes = [
            (0, 200, 'RGB inputs', ['Image / real driving video', 'Webcam code; device unverified']),
            (260, 200, 'Preprocessing', ['Bounded RGB frame', '640 x 192, float32, batch one']),
            (260, 100, 'Depth model', ['Monodepth2 / ResNet-18', 'Pretrained mono_640x192']),
            (0, 100, 'Relative depth', ['Numerical inverse depth + depth', 'Not calibrated distances in metres']),
            (0, 0, 'Visual output', ['RGB / depth / overlay', 'Preview + NPY + runtime logs']),
            (260, 0, 'Integrity and profiling', ['Pinned input / checkpoint hashes', 'Repeated predictions / timing']),
        ]
        for x, y, title, lines in boxes:
            c.setFillColor(colors.HexColor('#eff5f8'))
            c.setStrokeColor(TEAL)
            c.roundRect(x, y, 230, 77, 8, fill=1, stroke=1)
            c.setFillColor(NAVY)
            c.setFont('Helvetica-Bold', 11)
            c.drawString(x + 12, y + 55, title)
            c.setFont('Helvetica', 8.4)
            for i, line in enumerate(lines):
                c.drawString(x + 12, y + 34 - i * 14, line)
        c.setStrokeColor(TEAL)
        for x1, y1, x2, y2 in [(230, 238, 260, 238), (375, 200, 375, 177), (260, 138, 230, 138), (115, 100, 115, 77), (230, 38, 260, 38)]:
            c.line(x1, y1, x2, y2)
            if x2 != x1:
                direction = 1 if x2 > x1 else -1
                c.line(x2, y2, x2 - direction * 5, y2 + 3)
                c.line(x2, y2, x2 - direction * 5, y2 - 3)
            else:
                c.line(x2, y2, x2 - 3, y2 + 5)
                c.line(x2, y2, x2 + 3, y2 + 5)


def page_frame(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(TEAL)
    canvas.line(18 * mm, 280 * mm, 192 * mm, 280 * mm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 284 * mm, 'FINAL-YEAR PROJECT | IMPLEMENTATION RECORD')
    canvas.drawString(18 * mm, 12 * mm, 'Monocular depth estimation | Evidence-based progress report')
    canvas.drawRightString(192 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-date', default='2026-10-18')
    parser.add_argument('--output', default='output/pdf/implementation_progress_report.pdf')
    args = parser.parse_args()
    profile = load('runs/phase3_profile_final/summary.json')
    previous = load('runs/phase2_real_driving_rehearsal/validation.json')
    tests = load('runs/phase3_tests.json')
    inputs = load('data/demo/input_manifest.json')
    before = load('runs/phase3_render_before.json')
    after = load('runs/phase3_render_final.json')
    if profile['status'] != 'pass' or not tests['successful']:
        raise ValueError('The progress report requires passing profile and test evidence')
    base_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    story = []
    add = story.append

    add(p('Monocular Depth Estimation<br/>Implementation Progress Report', 'ReportTitle'))
    add(p(escape(args.report_date) + ' | Phases 1-3 | Hardware-adapted implementation', 'ReportSmall'))
    add(p('<b>Current result:</b> a working RGB-only depth-perception prototype with real driving-video replay, relative depth maps, colour overlays, saved outputs, portable input verification and repeatable performance profiling.'))
    add(p('The project addresses the absence of explicit depth information in ordinary camera images. A pretrained neural network estimates a dense relative-depth map from each RGB frame, providing a perception research prototype that runs on the available laptop hardware.'))
    add(table([
        ['Phase', 'Implementation position'],
        ['1 - Foundation', 'Image/video inference, RGB preprocessing, model loading and shared visualization implemented and tested.'],
        ['2 - Demo inputs', 'Real driving sample, retained license, image fallback, portable manifest and checksum verification implemented.'],
        ['3 - Stability / performance', 'Warmup-aware repeated profiling, numerical stability gate and measured rendering optimization implemented locally.'],
        ['4 - Accuracy evaluation', 'Metric formulas exist; full ground-truth dataset benchmarks remain pending.'],
        ['5 - Delivery', 'Setup and transfer instructions exist; final academic submission and presentation-machine acceptance remain pending.'],
    ], [43 * mm, 131 * mm]))
    add(Spacer(1, 12))
    add(p('<b>Scope:</b> the latest hardware-adaptation instruction relaxes the original PDF implementation constraints. This report records implemented and tested behavior, not completion of every original PDF deliverable.'))
    add(p('Model weights remain pretrained. No project-specific training, validated native CARLA camera run, calibrated metre-scale prediction or autonomous control is claimed.'))
    add(p('Source baseline: commit ' + escape(base_commit) + '; Phase 3 is the current working-tree update at report generation. Generated data and runtime evidence remain local.', 'ReportSmall'))

    add(PageBreak())
    add(p('1. Architecture and model', 'ReportHeading'))
    add(Architecture())
    add(Spacer(1, 8))
    add(p('The webcam and optional CARLA interfaces feed the same preprocessing and inference path, but their real hardware integrations remain unverified. The verified inputs are a saved RGB image and real driving-video replay.'))
    add(table([
        ['Component', 'Implemented contract'],
        ['Encoder', 'ResNet-18; encoder.pth; strict loading of official checkpoint parameters.'],
        ['Decoder', 'Monodepth2 multi-scale depth decoder; depth.pth; scale-zero output used for inference.'],
        ['Checkpoint', 'mono_640x192; KITTI-trained monocular baseline; immutable recorded hashes.'],
        ['Input', 'RGB HWC uint8 becomes a 1 x 3 x 192 x 640 float32 tensor in [0,1]. Original encoder normalization is retained.'],
        ['Output', 'Positive finite inverse-depth and relative-depth maps, 192x640. Depth is the reciprocal of inverse depth.'],
        ['Execution', 'Evaluation mode and torch.inference_mode; batch one; automatic CUDA/CPU selection.'],
    ], [37 * mm, 137 * mm]))
    add(Spacer(1, 10))
    add(p('Module locations: src/model.py, src/preprocess.py, src/visualize.py, src/metrics.py and optional src/carla_capture.py. Phase 2 adds src/demo_data.py; Phase 3 adds src/profiling.py. project.py is the shared command-line entry point.', 'ReportSmall'))

    add(PageBreak())
    add(p('2. Working features and input preparation', 'ReportHeading'))
    add(p('<b>Phase 1:</b> reproducible dependency pins; official source/checkpoint acquisition and provenance checks; saved-image prediction; video/webcam input adapters; RGB/depth/overlay output; first-frame numerical arrays and preview; runtime logging; optional simulator capture/replay tools.'))
    add(p('<b>Phase 2:</b> the prepare-demo command obtains a pinned Udacity driving sample and license. Every frame is decoded to check its count and dimensions, and a first-frame image fallback is saved. verify-demo validates the media, fallback image, baseline checkpoint and preprocessing identity after transfer to another laptop.'))
    add(table([
        ['Frozen demo input', 'Recorded value'],
        ['Source', 'Udacity / CarND-Advanced-Lane-Lines / project_video.mp4'],
        ['Repository revision', inputs['source_revision']],
        ['Video', f"{inputs['decoded_frames']} frames; {inputs['width']}x{inputs['height']}; {inputs['source_fps']:.0f} FPS; {inputs['source_duration_seconds']:.1f} source seconds"],
        ['License', 'Repository MIT license retained as UDACITY_LICENSE.txt.'],
        ['Ground truth', 'No aligned metric-depth ground truth is supplied by this sample.'],
    ], [42 * mm, 132 * mm]))
    add(Spacer(1, 12))
    preview = ROOT / 'runs/phase3_profile_final/preview.png'
    from PIL import Image as PILImage
    with PILImage.open(preview) as picture:
        aspect = picture.height / picture.width
    add(Image(str(preview), width=WIDTH, height=WIDTH * aspect))
    add(p('Actual profile preview from the driving clip: RGB, relative inverse depth and overlay. Colours visualize model predictions; they do not establish measured accuracy.', 'ReportSmall'))
    add(p('Demo outputs: preview.png, inverse_depth.npy, relative_depth.npy, runtime.csv, summary.json and metrics.json. Without ground truth, metrics.json is empty and the summary explicitly marks accuracy unavailable.'))
    add(p('Video replay is sequential. A processing cap below the native 25 FPS slows playback; it is not an interactive simulator or a guaranteed native-speed live stream.'))

    add(PageBreak())
    add(p('3. Phase 3: stability and performance', 'ReportHeading'))
    add(p(f"Profile hardware: <b>{escape(profile['gpu'])}</b>, Windows, Python {profile['python']}, PyTorch {profile['torch']}. Input 640x192, batch one and two CPU threads. Each of {profile['repeats']} repeats excludes {profile['warmup_frames_per_repeat']} full-pipeline warmup frames and measures the same {profile['measured_frames_per_repeat']} subsequent frames."))
    add(p('<b>Numerical stability:</b> every measured depth/inverse-depth pair is checked for shape, dtype, positivity, finiteness and reciprocal consistency. Both maps are hashed. All measured frame IDs and exact output fingerprints matched across the repeated runs on this machine.'))
    rows = [['Timing component', 'Median (ms)', 'P95 (ms)']]
    names = [('decode_ms', 'Decode / RGB conversion / resize'), ('model_ms', 'Model only'), ('processing_ms', 'Preprocessing + model + output transfer'), ('render_ms', 'Composite rendering'), ('validation_ms', 'Numerical checks / hashing'), ('pipeline_ms', 'Full measured headless cycle')]
    for key, label in names:
        item = profile['timings'][key]
        rows.append([label, f"{item['median']:.2f}", f"{item['p95']:.2f}"])
    add(table(rows, [112 * mm, 31 * mm, 31 * mm]))
    add(p('Processing includes model time: these rows overlap and must not be added together. The measured cycle excludes asset verification, setup, warmup, display, pacing and output-file writes.', 'ReportSmall'))
    add(table([['Repeat', 'Uncapped headless FPS', 'Measured frames']] + [[str(run['repeat']), f"{run['uncapped_headless_fps']:.2f}", str(run['measured_frames'])] for run in profile['runs']], [32 * mm, 100 * mm, 42 * mm]))
    add(Spacer(1, 10))
    improvement = (before['median_ms'] - after['median_ms']) / before['median_ms'] * 100
    add(p(f"<b>Rendering change:</b> cache the palette, font and static labels; blend the overlay with installed OpenCV. A separate warmed {before['measured_calls']}-call rendering check measured median {before['median_ms']:.2f} ms before and {after['median_ms']:.2f} ms after ({improvement:.1f}% lower in this observation). Timing varies with system load; this is not a universal speedup guarantee."))
    add(p('The colour palette and numerical predictions are preserved. Overlay rounding may differ by one uint8 colour level. No model weights, preprocessing normalization or training configuration were changed.'))

    add(PageBreak())
    add(p('4. Validation evidence and phase handoff', 'ReportHeading'))
    add(p(f"<b>{tests['tests_run']} tests passed</b>; failures: {tests['failures']}; errors: {tests['errors']}; skipped: {tests['skipped']}. The regression suite includes the original Phase 1 and Phase 2 checks plus Phase 3 profiling and visualization checks."))
    add(table([
        ['Area', 'Evidence covered'],
        ['Numerical / geometry', 'RGB order, tensor contract, depth decoding, intrinsics resizing, metric formulas and fixed colour scaling.'],
        ['Safety / integration', 'Archive traversal/size checks, checkpoint hashes, frame synchronization, replay validation and cleanup paths.'],
        ['Phase 2 transfer', 'Portable manifests, changed inputs/checkpoints, invalid metadata/paths, short decoding and source mismatch.'],
        ['Phase 3 stability', 'Timing percentiles, warmup exclusion, finite reciprocal maps, repeat mismatch rejection and palette/overlay fidelity.'],
        ['Actual model', 'Official-checkpoint repeatability and real driving-video CUDA profiling, separate from mocked tests.'],
    ], [40 * mm, 134 * mm]))
    add(Spacer(1, 10))
    add(p(f"<b>Earlier displayed rehearsal:</b> {previous['frames']} real-video frames over {previous['wall_seconds']:.2f} seconds at {previous['processed_fps']:.2f} FPS under a {previous['fps_limit']:.0f} FPS cap. This covers {previous['decoded_source_seconds_processed']:.0f} seconds of source footage. It is a different workload from the uncapped headless Phase 3 profile; the two FPS values are not directly comparable."))
    memory = profile['process_memory']
    add(p(f"Final-profile process peak working set: {memory['peak_working_set_mib']:.2f} MiB. PyTorch peak allocated GPU memory: {profile['pytorch_peak_allocated_mib']:.2f} MiB. Working set is not total system RAM or committed memory; PyTorch allocations are not total VRAM usage."))
    add(p('<b>Transition gate:</b> keep the accepted baseline and input hashes unchanged, run the full suite, verify-demo, image inference and a short video smoke test before accepting a later phase. Write into new output directories. A failed optional experiment must not replace the working baseline.'))
    add(p('Evidence locations (local, ignored): runs/phase3_tests.json; runs/phase3_profile_final/{summary.json,runtime.csv,fingerprints.json,preview.png}; runs/phase3_profile/summary.json (before rendering optimization); runs/phase3_render_before.json and runs/phase3_render_final.json.'))

    add(PageBreak())
    add(p('5. Remaining work and reproducible usage', 'ReportHeading'))
    add(table([
        ['Requirement', 'Honest current position'],
        ['Fine-tuning', 'Not performed. No prepared calibrated temporal training dataset or local pose/loss training pipeline; pretrained baseline retained.'],
        ['Ground-truth accuracy', 'No full KITTI/CARLA evaluation completed. Tested formulas do not prove AbsRel/RMSE targets.'],
        ['Live simulator / webcam', 'Native CARLA sensor validation and a physical-camera run remain pending.'],
        ['Presentation laptop', "RTX 3050 testing remains pending; current measurements belong to the local GTX 1650."],
        ['10+ FPS and metric scale', 'Uncapped headless FPS does not prove sustained displayed/live performance. Depth remains relative, without calibrated metre-scale output.'],
        ['Vehicle control', 'Path planning, steering and throttle control are outside the original PDF scope and are not implemented.'],
    ], [44 * mm, 130 * mm]))
    add(Spacer(1, 10))
    add(p('Run from the project directory in PowerShell (choose unused output names):'))
    code = r'''$depthPython = '.\.venv\Scripts\python.exe'
& $depthPython project.py assets
& $depthPython project.py prepare-demo
& $depthPython project.py verify-demo
& $depthPython -m unittest discover -s tests -v
& $depthPython project.py profile --device cuda --frames 60 `
  --warmup 5 --repeats 2 --output runs/profile_review_01
& $depthPython project.py demo --source video `
  --video data/demo/driving.mp4 --manifest data/demo/input_manifest.json `
  --device cuda --frames 350 --fps 5 --output runs/demo_review_01'''
    add(Preformatted(code, styles['ReportCode']))
    add(Spacer(1, 8))
    add(p('On the friend\'s laptop, transfer matching source, checkpoints, pinned third-party source and data/demo, then rebuild .venv. Verify CUDA, hashes and tests; rehearse the displayed demo there. Do not copy .venv or the CARLA runtime from the development machine.'))
    add(p('<b>References and provenance</b>', 'ReportSmall'))
    for title, url in [('Official Monodepth2 source and pretrained models', 'https://github.com/nianticlabs/monodepth2'), ('Udacity driving sample and repository license', 'https://github.com/udacity/CarND-Advanced-Lane-Lines'), ('Project repository', 'https://github.com/pjha26/FinalProject')]:
        add(p(f'<link href="{url}" color="#087e8b">{escape(title)}</link>', 'ReportSmall'))
    add(p('Original scope reference: monocular_depth_report_FINAL_FIXED (1).pdf, Sections 3.2-3.3 and Chapter 6. Current phase plans and repository code distinguish completed work from pending original targets.', 'ReportSmall'))

    doc = SimpleDocTemplate(str(output), pagesize=(210 * mm, 297 * mm), rightMargin=18 * mm, leftMargin=18 * mm,
                            topMargin=23 * mm, bottomMargin=20 * mm, title='Monocular Depth Estimation - Implementation Progress', author='Project implementation record')
    doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    from pypdf import PdfReader
    reader = PdfReader(output)
    print('Report:', output)
    print('Pages:', len(reader.pages))
    for index, page in enumerate(reader.pages, 1):
        if not (page.extract_text() or '').strip():
            raise ValueError('Empty report page: ' + str(index))


if __name__ == '__main__':
    main()
