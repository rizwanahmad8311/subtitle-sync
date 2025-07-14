import os
import tempfile
import json
import pandas as pd
import gradio as gr
from gradio.routes import mount_gradio_app
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
import traceback
import re
import webvtt
import uvicorn
import subprocess
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware



def wrap_text(text, max_line_length=29):
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        if len(' '.join(current_line + [word])) <= max_line_length:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
        
    return '\n'.join(lines)


def segment_text_file(input_content, output_path,):

    words = re.findall(r'\S+', input_content)
    if not words:
        return ""

    result = []
    current_line = ""

    for word in words:
        remaining_line = ""
        if len(current_line) + len(word) + 1 <= 58:
            current_line += word + " "
        else:
            if current_line:
                if '.' in current_line[29:]:
                    crr_line = current_line.split('.')
                    remaining_line = crr_line[-1].strip()
                    if len(crr_line) > 2:
                        current_line = ''.join([cr + "." for cr in crr_line[:-1]])
                    else:
                        current_line = crr_line[0].strip() + '.'

                # Check wrapped lines and extract excess if any
                wrapped = wrap_text(current_line).split('\n')
                result1 = '\n'.join(wrapped[2:])  
                if result1:
                    moved_word = result1
                    current_line = current_line.rstrip()
                    if current_line.endswith(moved_word):
                        current_line = current_line[:-(len(moved_word))].rstrip()

                    result.append(current_line.strip())
                    current_line = moved_word + " " + word + " "
                else:
                    result.append(current_line.strip())
                    current_line = remaining_line + " " + word + " "
            else:
                current_line = remaining_line + " " + word + " "

    if current_line:
        result.append(current_line.strip())

    # Write segmented output
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in result:
            f.write(seg.strip() + "\n")


def convert_to_srt(fragments):
    def format_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    srt_output = []
    index = 1
    for f in fragments:
        start = float(f.begin)
        end = float(f.end)
        text = f.text.strip()

        if end <= start or not text:
            continue

        lines = wrap_text(text)

        srt_output.append(f"{index}")
        srt_output.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        srt_output.append(lines)
        srt_output.append("")  # Empty line
        index += 1

    return "\n".join(srt_output)


def check_ffmpeg():
    """Check if FFmpeg is available on the system"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_video_file(file_path):
    """Check if the file is a video file based on extension"""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg'}
    return Path(file_path).suffix.lower() in video_extensions


def is_audio_file(file_path):
    """Check if the file is an audio file based on extension"""
    audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus'}
    return Path(file_path).suffix.lower() in audio_extensions


def convert_video_to_audio(video_path, output_path):
    """Convert video file to audio using FFmpeg"""
    try:
        # Use FFmpeg to extract audio from video
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',  # MP3 codec
            '-ab', '192k',  # Audio bitrate
            '-ar', '44100',  # Sample rate
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
        
        return True
    except Exception as e:
        raise RuntimeError(f"Error converting video to audio: {str(e)}")


def get_media_file_path(media_input):
    """Get file path from media input (audio or video)"""
    if media_input is None:
        return None
    
    if isinstance(media_input, str):
        return media_input
    elif isinstance(media_input, tuple) and len(media_input) >= 2:
        return media_input[1] if isinstance(media_input[1], str) else media_input[0]
    else:
        print(f"Debug: Unexpected media input type: {type(media_input)}")
        return str(media_input)


def get_text_file_path(text_input):
    if text_input is None:
        return None
    
    if isinstance(text_input, dict):
        return text_input['name']
    elif isinstance(text_input, str):
        return text_input
    else:
        print(f"Debug: Unexpected text input type: {type(text_input)}")
        return str(text_input)


def process_alignment(media_file, text_file, language, progress=gr.Progress()):
    
    if media_file is None:
        return "❌ Please upload an audio or video file", None, None, "", None, None
    
    if text_file is None:
        return "❌ Please upload a text file", None, None, "", None, None
    
    # Check if FFmpeg is available
    if not check_ffmpeg():
        return "❌ FFmpeg not found. Please install FFmpeg to process video files.", None, None, "", None, None
    
    # Initialize variables for cleanup
    temp_text_file_path = None
    temp_audio_file_path = None
    output_file = None
    
    try:
        progress(0.1, desc="Initializing...")
        
        # Create temporary directory for better file handling
        temp_dir = tempfile.mkdtemp()
        
        # Get the media file path
        media_file_path = get_media_file_path(media_file)
        if not media_file_path:
            raise ValueError("Could not determine media file path")
        
        # Verify media file exists
        if not os.path.exists(media_file_path):
            raise FileNotFoundError(f"Media file not found: {media_file_path}")
        
        # Process media file - convert video to audio if needed
        if is_video_file(media_file_path):
            progress(0.2, desc="Converting video to audio...")
            temp_audio_file_path = os.path.join(temp_dir, "extracted_audio.mp3")
            convert_video_to_audio(media_file_path, temp_audio_file_path)
            audio_file_path = temp_audio_file_path
            print(f"Debug: Video converted to audio: {audio_file_path}")
        elif is_audio_file(media_file_path):
            audio_file_path = media_file_path
            print(f"Debug: Using audio file directly: {audio_file_path}")
        else:
            raise ValueError("Unsupported file format. Please provide an audio or video file.")
        
        # Get the text file path
        text_file_path = get_text_file_path(text_file)
        if not text_file_path:
            raise ValueError("Could not determine text file path")
        
        print(f"Debug: Text file path: {text_file_path}")
        
        # Verify text file exists and read content
        if not os.path.exists(text_file_path):
            raise FileNotFoundError(f"Text file not found: {text_file_path}")
        
        # Read and validate text content
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(text_file_path, 'r', encoding='latin-1') as f:
                text_content = f.read().strip()
        
        if not text_content:
            raise ValueError("Text file is empty or contains only whitespace")
        
        progress(0.3, desc="Processing text file...")
        
        temp_text_file_path = os.path.join(temp_dir, "input_text.txt")
        segment_text_file(text_content, temp_text_file_path)
        
        # Verify temp text file was created
        if not os.path.exists(temp_text_file_path):
            raise RuntimeError("Failed to create temporary text file")
        
        # Create output file path
        output_file = os.path.join(temp_dir, "alignment_output.json")
        
        progress(0.4, desc="Creating task configuration...")
        
        # Create task configuration
        config_string = f"task_language={language}|is_text_type=plain|os_task_file_format=json"
        
        # Create and configure the task
        task = Task(config_string=config_string)
        
        # Set absolute paths
        task.audio_file_path_absolute = os.path.abspath(audio_file_path)
        task.text_file_path_absolute = os.path.abspath(temp_text_file_path)
        task.sync_map_file_path_absolute = os.path.abspath(output_file)
        
        progress(0.5, desc="Running alignment... This may take a while...")
        
        # Execute the alignment
        ExecuteTask(task).execute()
        
        progress(0.8, desc="Processing results...")
        
        # output sync map to file
        task.output_sync_map_file()

        # Check if output file was created
        if not os.path.exists(output_file):
            raise RuntimeError(f"Alignment output file was not created: {output_file}")
        
        # Read and process results
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # Read output and convert to SRT
        fragments = task.sync_map.fragments
        srt_content = convert_to_srt(fragments)

        srt_path = os.path.join(temp_dir, "output.srt")
        vtt_path = os.path.join(temp_dir, "output.vtt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        webvtt.from_srt(srt_path).save()
        
        if 'fragments' not in results or not results['fragments']:
            raise RuntimeError("No alignment fragments found in results")
        
        # Create DataFrame for display
        df_data = []
        for i, fragment in enumerate(results['fragments']):
            start_time = float(fragment['begin'])
            end_time = float(fragment['end'])
            duration = end_time - start_time
            text = fragment['lines'][0] if fragment['lines'] else ""
            
            df_data.append({
                'Segment': i + 1,
                'Start (s)': f"{start_time:.3f}",
                'End (s)': f"{end_time:.3f}",
                'Duration (s)': f"{duration:.3f}",
                'Text': text
            })
        
        df = pd.DataFrame(df_data)
        
        # Create summary
        total_duration = float(results['fragments'][-1]['end']) if results['fragments'] else 0
        avg_segment_length = total_duration / len(results['fragments']) if results['fragments'] else 0
        
        file_type = "video" if is_video_file(media_file_path) else "audio"
        
        summary = f"""
📊 **Alignment Summary**
- **Input type:** {file_type.title()} file
- **Total segments:** {len(results['fragments'])}
- **Total duration:** {total_duration:.3f} seconds
- **Average segment length:** {avg_segment_length:.3f} seconds
- **Language:** {language}
"""
        
        progress(1.0, desc="Complete!")
        
        print(f"Debug: Alignment completed successfully with {len(results['fragments'])} fragments")
        
        return (
            "✅ Alignment completed successfully!",
            df,
            output_file,  # For download
            summary,
            srt_path,
            vtt_path 
        )
            
    except Exception as e:
        print(f"Debug: Exception occurred: {str(e)}")
        print(f"Debug: Traceback: {traceback.format_exc()}")
        
        error_msg = f"❌ Error during alignment: {str(e)}\n\n"
        error_msg += "**Troubleshooting tips:**\n"
        error_msg += "- Ensure media file is in supported format (audio: WAV, MP3, FLAC, etc. | video: MP4, AVI, MKV, etc.)\n"
        error_msg += "- Ensure text file contains the spoken content\n"
        error_msg += "- Check that text file is in UTF-8 or Latin-1 encoding\n"
        error_msg += "- Verify both media and text files are not corrupted\n"
        error_msg += "- Try with a shorter audio/video/text pair first\n"
        error_msg += "- Make sure FFmpeg and Aeneas dependencies are properly installed\n"
        error_msg += "- For video files, ensure they contain audio tracks\n"
        
        if temp_text_file_path:
            error_msg += f"- Text file was processed from: {text_file_path}\n"
        
        error_msg += f"\n**Technical details:**\n```\n{traceback.format_exc()}\n```"
        
        return error_msg, None, None, "", None, None
    
    finally:
        # Clean up temporary files
        try:
            if temp_text_file_path and os.path.exists(temp_text_file_path):
                os.unlink(temp_text_file_path)
            if temp_audio_file_path and os.path.exists(temp_audio_file_path):
                os.unlink(temp_audio_file_path)
            print(f"Debug: Cleaned up temporary files")
        except Exception as cleanup_error:
            print(f"Debug: Error cleaning up temporary files: {cleanup_error}")


def create_interface():
    
    with gr.Blocks(title="Aeneas Forced Alignment Tool", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎯 Aeneas Forced Alignment Tool
        
        Upload an audio or video file and provide the corresponding text to generate precise time alignments.
        Perfect for creating subtitles, analyzing speech patterns, or preparing training data.
        
        **Supported formats:**
        - **Audio:** WAV, MP3, FLAC, AAC, OGG, WMA, M4A, OPUS
        - **Video:** MP4, AVI, MKV, MOV, WMV, FLV, WebM, M4V, 3GP, MPG, MPEG
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input Files")
                
                media_input = gr.File(
                    label="Audio or Video File",
                    file_types=[
                        ".wav", ".mp3", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".opus",  # Audio
                        ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".3gp", ".mpg", ".mpeg"  # Video
                    ],
                    file_count="single"
                )
                
                text_input = gr.File(
                    label="Text File (.txt)",
                    file_types=[".txt"],
                    file_count="single"
                )
                
                gr.Markdown("### ⚙️ Configuration")
                
                language_input = gr.Dropdown(
                    choices=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ar"],
                    value="en",
                    label="Language Code",
                    info="ISO language code (en=English, es=Spanish, etc.)"
                )
                
                process_btn = gr.Button("🚀 Process Alignment", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                status_output = gr.Markdown()
                summary_output = gr.Markdown()
                
                results_output = gr.Dataframe(
                    label="Alignment Results",
                    headers=["Segment", "Start (s)", "End (s)", "Duration (s)", "Text"],
                    datatype=["number", "str", "str", "str", "str"],
                    interactive=False
                )
                
                download_output = gr.File(
                    label="Download JSON Results",
                    visible=False
                )

                srt_file_output = gr.File(
                    label="Download SRT File",
                    visible=False
                )

                vtt_file_output = gr.File(
                    label="Download VTT File",
                    visible=False
                )
        
        # Event handlers
        process_btn.click(
            fn=process_alignment,
            inputs=[
                media_input,
                text_input,
                language_input,
            ],
            outputs=[
                status_output,
                results_output,
                download_output,
                summary_output,
                srt_file_output,
                vtt_file_output
            ]
        ).then(
            fn=lambda x: gr.update(visible=x is not None),
            inputs=download_output,
            outputs=download_output
        ).then(
            fn=lambda x: gr.update(visible=x is not None),
            inputs=srt_file_output,
            outputs=srt_file_output
        ).then(
            fn=lambda x: gr.update(visible=x is not None),
            inputs=vtt_file_output,
            outputs=vtt_file_output
        )
    
    return interface


fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.post("/align/")
async def align_api(
    media_file: UploadFile = File(...),
    text_file: UploadFile = File(...),
    language: str = Form(default="en")
):
    try:
        # Validate text file
        if not text_file.filename.endswith(".txt"):
            raise HTTPException(
                status_code=400,
                detail="Text file must be a .txt file"
            )
        
        # Check if media file is supported
        media_filename = media_file.filename.lower()
        audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus'}
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg'}
        
        file_ext = Path(media_filename).suffix.lower()
        if file_ext not in audio_extensions and file_ext not in video_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported media file format: {file_ext}. Supported formats: {', '.join(sorted(audio_extensions | video_extensions))}"
            )
        
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_media:
            shutil.copyfileobj(media_file.file, temp_media)
            media_path = temp_media.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w+', encoding='utf-8') as temp_text:
            content = (await text_file.read()).decode('utf-8', errors='ignore')
            temp_text.write(content)
            temp_text.flush()
            text_path = temp_text.name

        # Process alignment
        status, df, json_path, summary, srt_path, vtt_path = process_alignment(media_path, text_path, language)

        # Clean up uploaded files
        try:
            os.unlink(media_path)
            os.unlink(text_path)
        except Exception as cleanup_error:
            print(f"Warning: Error cleaning up uploaded files: {cleanup_error}")

        if "Error" in status or status.startswith("❌"):
            raise HTTPException(status_code=500, detail=status)

        response = {
            "status": status,
            "summary": summary,
            "segments": df.to_dict(orient="records") if df is not None else [],
            "download_links": {
                "alignment_json": json_path,
                "srt": srt_path,
                "vtt": vtt_path
            }
        }

        return JSONResponse(status_code=200, content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {str(e)}"
        )


interface = create_interface()

app = mount_gradio_app(fastapi_app, interface, path="")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)