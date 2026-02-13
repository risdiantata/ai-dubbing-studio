import os
import sys
import time
import subprocess
import threading
import uuid
import shutil
import asyncio
import math
import random
import re
import numpy as np
from flask import Flask, request, jsonify, send_file, session, redirect, render_template_string
from flask_cors import CORS
import edge_tts
import librosa
import speech_recognition as sr
from deep_translator import GoogleTranslator
import moviepy.editor as mp
from moviepy.editor import vfx, CompositeAudioClip, CompositeVideoClip, concatenate_videoclips
import PIL.Image

if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

app = Flask(__name__)
CORS(app)
app.secret_key = 'dubbing-studio-pro-2025-secret'

# ==========================================
# USER ACCOUNTS (username: {password, role})
# ==========================================
USERS = {
    'admin': {'password': 'admin123', 'role': 'pro'},
    'pro': {'password': 'pro123', 'role': 'pro'},
}

# Demo limits
DEMO_MAX_DURATION = 30  # seconds

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

tasks = {}

# VOICE MAPPING
VOICE_MAP = {
    'id-ID': {'male': 'id-ID-ArdiNeural', 'female': 'id-ID-GadisNeural'},
    'en-US': {'male': 'en-US-GuyNeural', 'female': 'en-US-JennyNeural'},
    'ja-JP': {'male': 'ja-JP-KeitaNeural', 'female': 'ja-JP-NanamiNeural'},
    'ko-KR': {'male': 'ko-KR-InJoonNeural', 'female': 'ko-KR-SunHiNeural'},
    'zh-CN': {'male': 'zh-CN-YunxiNeural', 'female': 'zh-CN-XiaoxiaoNeural'},
    'su-ID': {'male': 'su-ID-JajangNeural', 'female': 'su-ID-TutiNeural'},
    'de-DE': {'male': 'de-DE-KillianNeural', 'female': 'de-DE-KatjaNeural'},
    'es-ES': {'male': 'es-ES-AlvaroNeural', 'female': 'es-ES-ElviraNeural'},
    'pt-BR': {'male': 'pt-BR-AntonioNeural', 'female': 'pt-BR-FranciscaNeural'},
    'ru-RU': {'male': 'ru-RU-DmitryNeural', 'female': 'ru-RU-SvetlanaNeural'},
    'hi-IN': {'male': 'hi-IN-MadhurNeural', 'female': 'hi-IN-SwaraNeural'},
    'th-TH': {'male': 'th-TH-NiwatNeural', 'female': 'th-TH-PremwadeeNeural'},
    'vi-VN': {'male': 'vi-VN-NamMinhNeural', 'female': 'vi-VN-HoaiMyNeural'},
    'ar-SA': {'male': 'ar-SA-HamedNeural', 'female': 'ar-SA-ZariyahNeural'},
    'tr-TR': {'male': 'tr-TR-AhmetNeural', 'female': 'tr-TR-EmelNeural'},
    'it-IT': {'male': 'it-IT-DiegoNeural', 'female': 'it-IT-ElsaNeural'},
    'ms-MY': {'male': 'ms-MY-OsmanNeural', 'female': 'ms-MY-YasminNeural'},
    'fil-PH': {'male': 'fil-PH-AngeloNeural', 'female': 'fil-PH-BlessicaNeural'},
}

# EFFECTS MAPPING
EFFECTS_MAP = {
    'normal': {'rate': '+0%', 'pitch': '+0Hz'},
    'happy': {'rate': '+10%', 'pitch': '+2Hz'},
    'sad': {'rate': '-10%', 'pitch': '-2Hz'},
    'angry': {'rate': '+5%', 'pitch': '+5Hz'},
    'terify': {'rate': '-20%', 'pitch': '-5Hz'},
    'chipmunk': {'rate': '+0%', 'pitch': '+10Hz'},
    'monster': {'rate': '-10%', 'pitch': '-10Hz'},
    'story': {'rate': '-10%', 'pitch': '-2Hz'},
    'dialog': {'rate': '+0%', 'pitch': '+0Hz'},
    'news': {'rate': '+5%', 'pitch': '+2Hz'},
    'kid': {'rate': '+10%', 'pitch': '+15Hz'},
}

def cleanup_temp(paths):
    for p in paths:
        try:
            if p and os.path.exists(p): os.remove(p)
        except Exception: pass

def fmt_timestamp(t):
    """Format seconds to HH:MM:SS.mmm"""
    m, s = divmod(max(0, t), 60)
    h, m = divmod(m, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02}.{ms:03}"

async def generate_tts_edge(text, voice, output_file, rate="+0%", pitch="+0Hz"):
    """Generate TTS audio and VTT subtitles using Edge TTS"""
    base_name = output_file.rsplit('.', 1)[0]
    vtt_file = base_name + ".vtt"
    
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    submaker = edge_tts.SubMaker()
    
    with open(output_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.feed(chunk)
                
    # Generate SRT
    try:
        srt_content = submaker.get_srt()
    except:
        srt_content = ""
    
    # FALLBACK: If SRT is empty, generate a simple one
    if not srt_content or len(srt_content) < 10:
        print(f"   -> Warning: No timestamps from edge-tts. Generating fallback VTT...")
        srt_content = "1\n00:00:00,000 --> 00:00:10,000\n" + text + "\n"
        
    # Save VTT (Convert SRT to VTT)
    vtt_content = "WEBVTT\n\n" + srt_content.replace(",", ".")
    
    with open(vtt_file, "w", encoding="utf-8") as file:
        file.write(vtt_content)
        
    return vtt_file

def detect_gender(audio_path):
    try:
        y, sr_rate = librosa.load(audio_path, sr=None)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr_rate)
        threshold = np.median(magnitudes)
        index = magnitudes > threshold
        pitch = pitches[index]
        if len(pitch) == 0: return 'female'
        pitch = pitch[pitch > 50] 
        avg_pitch = np.mean(pitch)
        gender_res = 'male' if avg_pitch < 165 else 'female'
        print(f"   -> AI Pitch: {avg_pitch:.2f} Hz => {gender_res.upper()}")
        return gender_res
    except: return 'female'

def detect_gender_from_array(y_segment, sr_rate):
    """Detect gender from a numpy audio array segment using pitch analysis"""
    try:
        if len(y_segment) < sr_rate * 0.3:  # Too short
            return 'female'
        pitches, magnitudes = librosa.piptrack(y=y_segment, sr=sr_rate)
        threshold = np.median(magnitudes)
        index = magnitudes > threshold
        pitch = pitches[index]
        if len(pitch) == 0: return 'female'
        pitch = pitch[pitch > 50]
        if len(pitch) == 0: return 'female'
        avg_pitch = np.mean(pitch)
        return 'male' if avg_pitch < 165 else 'female'
    except:
        return 'female'

def detect_language_audio(audio_path):
    """Tries to detect language by transcribing first 15s with multiple langs"""
    r = sr.Recognizer()
    candidates = ['id-ID', 'en-US', 'ja-JP', 'ko-KR', 'zh-CN', 'zh-TW', 'zh', 'su-ID', 'fr-FR', 'de-DE', 'es-ES', 'ru-RU', 'hi-IN', 'ar-SA', 'pt-BR']
    scores = {}
    
    with sr.AudioFile(audio_path) as source:
        audio_data = r.record(source, duration=15)
        
        for lang in candidates:
            try:
                l = lang
                if lang == 'zh': l = 'zh-CN' 
                text = r.recognize_google(audio_data, language=l)
                score = len(text)
                if any(x in lang for x in ['zh', 'ja', 'ko']):
                    score *= 3
                scores[lang] = score
            except:
                scores[lang] = 0
                
    if not scores: return 'en-US'
    best_lang = max(scores, key=scores.get)
    print(f"   -> Auto-Detected Language: {best_lang} (Score: {scores[best_lang]})")
    if best_lang == 'zh': best_lang = 'zh-CN'
    return best_lang if scores[best_lang] > 0 else 'en-US'

def transcribe_large_audio(audio_path, language, chunk_len=60):
    """Transcribe audio without timestamps (legacy, for fallback)"""
    r = sr.Recognizer()
    full_text = []
    with sr.AudioFile(audio_path) as source:
        total_duration = source.DURATION
        processed = 0
        print(f"   -> Processing {total_duration:.2f}s in chunks...")
        while processed < total_duration:
            audio_data = r.record(source, duration=chunk_len)
            try:
                text = r.recognize_google(audio_data, language=language)
                full_text.append(text)
            except: pass
            processed += chunk_len
    return " ".join(full_text)

def detect_speech_segments(audio_path, min_segment_sec=1.0, merge_gap_sec=0.5):
    """
    Detect speech segments using librosa silence detection.
    Returns list of (start_sec, end_sec) tuples where speech occurs.
    """
    try:
        y, sr_rate = librosa.load(audio_path, sr=16000, mono=True)
        total_dur = len(y) / sr_rate
        
        # Detect non-silent intervals
        intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
        
        if len(intervals) == 0:
            print(f"   -> No speech detected, using full duration")
            return [(0.0, total_dur)]
        
        # Convert sample indices to seconds
        segments = [(s / sr_rate, e / sr_rate) for s, e in intervals]
        
        # Merge close segments (< merge_gap_sec apart)
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < merge_gap_sec:
                merged[-1] = (prev_start, end)  # Merge
            else:
                merged.append((start, end))
        
        # Filter out very short segments
        merged = [(s, e) for s, e in merged if (e - s) >= min_segment_sec]
        
        if not merged:
            merged = [(0.0, total_dur)]
        
        print(f"   -> Detected {len(merged)} speech segments in {total_dur:.1f}s audio")
        for i, (s, e) in enumerate(merged):
            print(f"      Segment {i+1}: {s:.1f}s - {e:.1f}s ({e-s:.1f}s)")
        
        return merged
    except Exception as e:
        print(f"   -> Speech detection failed: {e}, using even distribution")
        try:
            y, sr_rate = librosa.load(audio_path, sr=16000, duration=1)
            total_dur = librosa.get_duration(filename=audio_path)
        except:
            total_dur = 60
        return [(0.0, total_dur)]

def transcribe_with_timestamps(audio_path, language, segments):
    """
    Transcribe each speech segment individually.
    Returns list of dicts: [{start, end, text}, ...]
    """
    import soundfile as sf
    import tempfile
    
    r = sr.Recognizer()
    results = []
    
    # Load full audio once
    try:
        y, sr_rate = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        print(f"   -> Cannot load audio for segment transcription: {e}")
        # Fallback: transcribe entire file
        text = transcribe_large_audio(audio_path, language)
        total_dur = 60
        try:
            with sr.AudioFile(audio_path) as src:
                total_dur = src.DURATION
        except: pass
        return [{'start': 0, 'end': total_dur, 'text': text}]
    
    for i, (seg_start, seg_end) in enumerate(segments):
        start_sample = int(seg_start * sr_rate)
        end_sample = int(seg_end * sr_rate)
        segment_audio = y[start_sample:end_sample]
        
        if len(segment_audio) < sr_rate * 0.3:  # Skip segments < 0.3s
            continue
        
        # Write segment to temp file
        temp_path = os.path.join(UPLOAD_FOLDER, f"_seg_tmp_{i}.wav")
        try:
            sf.write(temp_path, segment_audio, sr_rate)
            
            with sr.AudioFile(temp_path) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language=language)
                if text and text.strip():
                    # Detect gender for this segment
                    seg_gender = detect_gender_from_array(segment_audio, sr_rate)
                    results.append({
                        'start': seg_start,
                        'end': seg_end,
                        'text': text.strip(),
                        'gender': seg_gender
                    })
                    print(f"      [{i+1}/{len(segments)}] {seg_start:.1f}s-{seg_end:.1f}s [{seg_gender[0].upper()}]: \"{text[:35]}...\"")
        except sr.UnknownValueError:
            print(f"      [{i+1}/{len(segments)}] {seg_start:.1f}s-{seg_end:.1f}s: (no speech)")
        except Exception as e:
            print(f"      [{i+1}/{len(segments)}] {seg_start:.1f}s-{seg_end:.1f}s: Error: {e}")
        finally:
            try: os.remove(temp_path)
            except: pass
    
    if not results:
        # Fallback to full transcription
        text = transcribe_large_audio(audio_path, language)
        total_dur = len(y) / sr_rate
        results = [{'start': 0, 'end': total_dur, 'text': text}]
    
    return results

def translate_segments(segments, source_lang, target_lang):
    """Translate each segment's text, preserving timestamps"""
    if source_lang == target_lang:
        return segments
    
    try:
        src_code = source_lang.split('-')[0] if 'zh' not in source_lang else source_lang
        tgt_code = target_lang.split('-')[0] if 'zh' not in target_lang else target_lang
        if tgt_code == 'zh': tgt_code = 'zh-CN'
        s_code = src_code if src_code else 'auto'
        trans = GoogleTranslator(source=s_code, target=tgt_code)
        
        translated = []
        for seg in segments:
            try:
                t_text = trans.translate(seg['text'])
                translated.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': t_text if t_text else seg['text'],
                    'original': seg['text'],
                    'gender': seg.get('gender', 'female')
                })
            except:
                translated.append({**seg, 'original': seg['text']})
        return translated
    except:
        return [{**s, 'original': s['text']} for s in segments]

def translate_large_text(text, translator):
    if not text: return ""
    limit = 4500
    chunks = [text[i:i+limit] for i in range(0, len(text), limit)]
    translated = []
    for chunk in chunks:
        try:
            translated.append(translator.translate(chunk))
        except: translated.append(chunk)
    return " ".join(translated)

def split_text_to_sentences(text):
    """Split text into sentences for subtitle sync"""
    # Split by . ? ! but keep the delimiter
    raw_sentences = re.split(r'(?<=[.?!])\s+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip() and len(s.strip()) > 1]
    if not sentences:
        sentences = [text]
    return sentences

def generate_synced_vtt(sentences, video_duration):
    """Generate VTT subtitle content with timestamps distributed across video duration"""
    if not sentences or video_duration <= 0:
        return ""
    
    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        total_chars = 1
    
    vtt = "WEBVTT\n\n"
    current_time = 0.0
    
    for i, sentence in enumerate(sentences):
        # Weight by character count
        char_ratio = len(sentence) / total_chars
        allocated_time = max(char_ratio * video_duration, 1.5)  # Min 1.5s per subtitle
        
        # Don't exceed video duration
        if current_time + allocated_time > video_duration:
            allocated_time = video_duration - current_time
        
        if allocated_time <= 0:
            break
            
        start = current_time
        end = current_time + allocated_time
        
        vtt += f"{i+1}\n{fmt_timestamp(start)} --> {fmt_timestamp(end)}\n{sentence}\n\n"
        current_time = end
    
    return vtt

def merge_short_sentences(sentences, max_segments=50):
    """Merge sentences into batches if there are too many, to reduce TTS calls for long videos"""
    if len(sentences) <= max_segments:
        return sentences
    
    # Calculate how many sentences per batch
    batch_size = math.ceil(len(sentences) / max_segments)
    merged = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        merged.append(' '.join(batch))
    
    print(f"   -> Merged {len(sentences)} sentences into {len(merged)} batches")
    return merged

def generate_synced_audio(task_id, sentences, voice, video_duration, base_rate, base_pitch):
    """
    Generate TTS audio segments for each sentence, synced to video duration.
    Returns: path to the combined audio file, and VTT content string.
    Updates tasks[task_id] progress from 70% to 89% during generation.
    """
    global tasks
    
    if not sentences or video_duration <= 0:
        return None, ""
    
    # Merge sentences if too many (for long videos)
    sentences = merge_short_sentences(sentences, max_segments=50)
    
    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        total_chars = 1
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    audio_clips = []
    vtt_entries = []
    current_time = 0.0
    temp_seg_files = []
    num_sentences = len(sentences)
    
    try:
        for i, sentence in enumerate(sentences):
            # Update progress (70% to 89%)
            progress_pct = 70 + int((i / num_sentences) * 19)
            tasks[task_id]['percent'] = progress_pct
            tasks[task_id]['status'] = f'Membuat Suara {i+1}/{num_sentences}...'
            
            # Calculate allocated time for this sentence (based on character weight)
            char_ratio = len(sentence) / total_chars
            allocated_time = max(char_ratio * video_duration, 1.0)  # Min 1.0s
            
            # Don't exceed remaining video duration
            remaining = video_duration - current_time
            if remaining <= 0:
                break
            allocated_time = min(allocated_time, remaining)
            
            seg_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_seg_{i}.mp3")
            temp_seg_files.append(seg_path)
            
            try:
                # Generate TTS for this sentence
                loop.run_until_complete(generate_tts_edge(sentence, voice, seg_path, rate=base_rate, pitch=base_pitch))
                
                if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                    seg_clip = mp.AudioFileClip(seg_path)
                    seg_dur = seg_clip.duration
                    
                    # If audio is much longer than allocated, speed it up (max 1.5x)
                    if seg_dur > allocated_time * 1.1:
                        speed_factor = min(seg_dur / allocated_time, 1.5)
                        pct = int((speed_factor - 1) * 100)
                        rate_str = f"+{pct}%"
                        print(f"   -> Sentence {i+1}: Speeding up by {rate_str} ({seg_dur:.1f}s -> {allocated_time:.1f}s)")
                        seg_clip.close()
                        
                        # Re-generate with faster rate
                        loop.run_until_complete(generate_tts_edge(sentence, voice, seg_path, rate=rate_str, pitch=base_pitch))
                        seg_clip = mp.AudioFileClip(seg_path)
                        seg_dur = seg_clip.duration
                    
                    # Set start time for CompositeAudioClip positioning
                    seg_clip = seg_clip.set_start(current_time)
                    audio_clips.append(seg_clip)
                    
                    # VTT entry: text visible for allocated time
                    actual_end = min(current_time + allocated_time, video_duration)
                    vtt_entries.append(
                        f"{i+1}\n{fmt_timestamp(current_time)} --> {fmt_timestamp(actual_end)}\n{sentence}\n\n"
                    )
                    
                    print(f"   -> Sentence {i+1}/{num_sentences}: '{sentence[:30]}...' @ {current_time:.1f}s (dur={seg_dur:.1f}s, alloc={allocated_time:.1f}s)")
                    
                    current_time += allocated_time
                    
            except Exception as e:
                print(f"   -> Segment {i+1} failed: {e}")
                current_time += allocated_time  # Skip ahead anyway
        
        # Compile audio
        tasks[task_id]['status'] = 'Menggabungkan Audio...'
        combined_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_tts.mp3")
        vtt_content = "WEBVTT\n\n" + "".join(vtt_entries)
        
        if audio_clips:
            full_audio = CompositeAudioClip(audio_clips)
            # Set duration to match video
            full_audio = full_audio.set_duration(video_duration)
            full_audio.write_audiofile(combined_path, fps=24000, logger=None)
            
            # Close clips to free memory
            for clip in audio_clips:
                try: clip.close()
                except: pass
            try: full_audio.close()
            except: pass
            
            print(f"[{task_id}] Synced audio compiled: {len(audio_clips)} segments over {video_duration:.1f}s")
        else:
            # Create silent audio as fallback
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except:
                ffmpeg_exe = 'ffmpeg'
            cmd = [ffmpeg_exe, '-f', 'lavfi', '-i', f'anullsrc=r=24000:cl=mono', '-t', str(video_duration), '-q:a', '9', '-acodec', 'libmp3lame', combined_path, '-y']
            subprocess.run(cmd, timeout=60)
            print(f"[{task_id}] Warning: No audio segments, using silence")
            
    finally:
        loop.close()
        # Cleanup segment files
        for f in temp_seg_files:
            try:
                if os.path.exists(f): os.remove(f)
                vtt_f = f.replace('.mp3', '.vtt')
                if os.path.exists(vtt_f): os.remove(vtt_f)
            except: pass
    
    return combined_path, vtt_content

def generate_speech_synced_audio(task_id, translated_segments, voice, video_duration, base_rate, base_pitch, voice_map=None, target_lang=None):
    """
    Generate TTS audio for each translated segment, placed at the ORIGINAL speech timestamps.
    Uses per-segment gender detection to match male/female voices.
    
    translated_segments: list of {start, end, text, original, gender}
    voice_map: dict like {'male': 'id-ID-ArdiNeural', 'female': 'id-ID-GadisNeural'}
    Returns: path to combined audio file, VTT content string
    """
    global tasks
    
    if not translated_segments or video_duration <= 0:
        return None, ""
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    audio_clips = []
    vtt_entries = []
    temp_seg_files = []
    num_segs = len(translated_segments)
    
    try:
        for i, seg in enumerate(translated_segments):
            seg_start = seg['start']
            seg_end = seg['end']
            seg_text = seg['text']
            allocated_time = seg_end - seg_start
            
            if allocated_time < 0.3 or not seg_text.strip():
                continue
            
            # Select voice based on segment gender
            seg_gender = seg.get('gender', 'female')
            if voice_map and target_lang:
                seg_voice = voice_map.get(seg_gender, voice)
            else:
                seg_voice = voice
            
            # Update progress (70% to 89%)
            progress_pct = 70 + int((i / num_segs) * 19)
            tasks[task_id]['percent'] = progress_pct
            tasks[task_id]['status'] = f'Dubbing {i+1}/{num_segs} [{seg_gender[0].upper()}]...'
            
            seg_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_seg_{i}.mp3")
            temp_seg_files.append(seg_path)
            
            try:
                # Generate TTS for this segment with gender-matched voice
                loop.run_until_complete(generate_tts_edge(seg_text, seg_voice, seg_path, rate=base_rate, pitch=base_pitch))
                
                if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                    seg_clip = mp.AudioFileClip(seg_path)
                    seg_dur = seg_clip.duration
                    
                    # If TTS audio is longer than the allocated time slot, speed it up
                    if seg_dur > allocated_time * 1.15:
                        speed_factor = min(seg_dur / allocated_time, 1.6)
                        pct = int((speed_factor - 1) * 100)
                        rate_str = f"+{pct}%"
                        print(f"   -> Seg {i+1}: Speed up {rate_str} ({seg_dur:.1f}s -> {allocated_time:.1f}s)")
                        seg_clip.close()
                        
                        # Re-generate with faster rate
                        loop.run_until_complete(generate_tts_edge(seg_text, seg_voice, seg_path, rate=rate_str, pitch=base_pitch))
                        seg_clip = mp.AudioFileClip(seg_path)
                        seg_dur = seg_clip.duration
                    
                    # Place audio at the EXACT original speech timestamp
                    seg_clip = seg_clip.set_start(seg_start)
                    audio_clips.append(seg_clip)
                    
                    # Subtitle also at the exact timestamp
                    sub_end = min(seg_start + max(seg_dur, allocated_time), video_duration)
                    vtt_entries.append(
                        f"{i+1}\n{fmt_timestamp(seg_start)} --> {fmt_timestamp(sub_end)}\n{seg_text}\n\n"
                    )
                    
                    print(f"   -> Seg {i+1}/{num_segs} [{seg_gender[0].upper()}]: @{seg_start:.1f}s-{seg_end:.1f}s \"{seg_text[:30]}...\" (tts={seg_dur:.1f}s)")
                    
            except Exception as e:
                print(f"   -> Seg {i+1} TTS failed: {e}")
        
        # Compile all audio clips
        tasks[task_id]['status'] = 'Menggabungkan Audio...'
        combined_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_tts.mp3")
        vtt_content = "WEBVTT\n\n" + "".join(vtt_entries)
        
        if audio_clips:
            full_audio = CompositeAudioClip(audio_clips)
            full_audio = full_audio.set_duration(video_duration)
            full_audio.write_audiofile(combined_path, fps=24000, logger=None)
            
            for clip in audio_clips:
                try: clip.close()
                except: pass
            try: full_audio.close()
            except: pass
            
            print(f"[{task_id}] Speech-synced audio: {len(audio_clips)} segments over {video_duration:.1f}s")
        else:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except:
                ffmpeg_exe = 'ffmpeg'
            cmd = [ffmpeg_exe, '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono', '-t', str(video_duration), '-q:a', '9', '-acodec', 'libmp3lame', combined_path, '-y']
            subprocess.run(cmd, timeout=60)
            print(f"[{task_id}] Warning: No audio segments, using silence")
            
    finally:
        loop.close()
        for f in temp_seg_files:
            try:
                if os.path.exists(f): os.remove(f)
                vtt_f = f.replace('.mp3', '.vtt')
                if os.path.exists(vtt_f): os.remove(vtt_f)
            except: pass
    
    return combined_path, vtt_content


def process_dubbing(task_id, file_path, source_lang, target_lang, video_format, gender, 
                    enhance_video, unique_mode, manual_text, voice_effect, app_host, 
                    mode='video', backsound_path=None, animation_mode='none', 
                    compress_level='medium', keep_original='false', burn_subtitles='true', 
                    subtitle_lang='match', dub_audio_path=None, sync_mode='audio_to_video'):
    """Main dubbing pipeline"""
    global tasks
    is_image = (mode == 'image')
    is_tts = (mode == 'tts')
    temp_files = [file_path]
    if backsound_path: temp_files.append(backsound_path)
    if dub_audio_path: temp_files.append(dub_audio_path)
    
    # ==========================================
    # IMAGE TO VIDEO MODE
    # ==========================================
    if mode == 'image':
        try:
            tasks[task_id] = {'percent': 10, 'status': 'Preparing Image...', 'result': None}
            if not manual_text: raise ValueError("Mode Gambar butuh teks!")
            
            # Translate
            translated_text = manual_text
            if source_lang != target_lang and source_lang != 'auto':
                tasks[task_id]['percent'] = 30
                tasks[task_id]['status'] = 'Translating...'
                try:
                    src_code = source_lang.split('-')[0] if 'zh' not in source_lang else source_lang
                    tgt_code = target_lang.split('-')[0] if 'zh' not in target_lang else target_lang
                    if tgt_code == 'zh': tgt_code = 'zh-CN'
                    trans = GoogleTranslator(source=src_code, target=tgt_code)
                    translated_text = translate_large_text(manual_text, trans)
                except: pass

            # Generate Voice
            tasks[task_id]['percent'] = 50
            tasks[task_id]['status'] = f'Generating Voice ({gender})...'
            
            tts_filename = f"{task_id}_tts.mp3"
            tts_path = os.path.join(UPLOAD_FOLDER, tts_filename)
            
            if gender == 'auto': gender = 'female'
            voice = VOICE_MAP.get(target_lang, {}).get(gender, 'en-US-JennyNeural')
            effect_params = EFFECTS_MAP.get(voice_effect, EFFECTS_MAP['normal'])
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(generate_tts_edge(translated_text, voice, tts_path, rate=effect_params['rate'], pitch=effect_params['pitch']))
            loop.close()
            
            # Prepare Audio
            final_audio = mp.AudioFileClip(tts_path)
            
            if backsound_path and os.path.exists(backsound_path):
                print(f"[{task_id}] Mixing Backsound...")
                try:
                    bg_audio = mp.AudioFileClip(backsound_path)
                    target_dur = final_audio.duration
                    if bg_audio.duration < target_dur:
                        n = int(target_dur / bg_audio.duration) + 2
                        bg_audio = mp.concatenate_audioclips([bg_audio] * n)
                    safe_dur = min(target_dur, bg_audio.duration - 0.01)
                    bg_audio = bg_audio.subclip(0, safe_dur).volumex(0.25)
                    final_audio = CompositeAudioClip([bg_audio, final_audio])
                except Exception as e: print(f"Mix Error: {e}")

            # Create Video from Image
            tasks[task_id]['percent'] = 80
            tasks[task_id]['status'] = 'Rendering Video...'
            
            duration = final_audio.duration + 1
            img_clip = mp.ImageClip(file_path).set_duration(duration).set_fps(24)
            
            final_video = img_clip
            w, h = img_clip.size
            
            if animation_mode == 'cinematic':
                zoomed = img_clip.resize(lambda t: 1 + 0.08 * t / duration)
                final_video = mp.CompositeVideoClip([zoomed.set_position('center')], size=(w,h))
            elif animation_mode == 'breathing':
                zoomed = img_clip.resize(lambda t: 1 + 0.02 * math.sin(t * 2))
                final_video = mp.CompositeVideoClip([zoomed.set_position('center')], size=(w,h))
            elif animation_mode == 'pan':
                zoomed = img_clip.resize(lambda t: 1 + 0.2 * t / duration)
                final_video = mp.CompositeVideoClip([zoomed.set_position('center')], size=(w,h))
            
            # Pad audio to match video duration
            if final_audio.duration < duration:
                final_audio = CompositeAudioClip([final_audio]).set_duration(duration)
            
            final_video = final_video.set_audio(final_audio)
            
            if enhance_video == 'true' or unique_mode == 'true':
                final_video = final_video.fx(vfx.colorx, 1.2)

            # Format
            if video_format == 'landscape': 
                final_video = final_video.resize(height=720)
            elif video_format == 'portrait':
                w, h = final_video.size
                if w > h:
                    new_w = h * (9/16)
                    x1 = (w/2) - (new_w/2)
                    final_video = final_video.crop(x1=x1, y1=0, width=new_w, height=h)

            out_name = f"image_vid_{task_id}.mp4"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            step1_file = os.path.join(UPLOAD_FOLDER, f"{task_id}_step1.mp4")
            final_video.write_videofile(step1_file, codec='libx264', audio_codec='aac', fps=24, threads=1, preset='ultrafast')
            
            try: final_audio.close()
            except: pass

            # Subtitles (Burn-in)
            vtt_path = tts_path.replace('.mp3', '.vtt')
            final_output_path = step1_file
            
            if os.path.exists(vtt_path) and burn_subtitles == 'true':
                print(f"[{task_id}] Burning Subtitles...")
                try:
                    vtt_path_safe = vtt_path.replace('\\', '/')
                    
                    try:
                        import imageio_ffmpeg
                        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    except:
                        ffmpeg_exe = 'ffmpeg'
                    
                    style = "Fontname=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000,BorderStyle=4,Outline=1,Shadow=0,Alignment=2,MarginV=25"
                    cmd = [
                        ffmpeg_exe, '-y', 
                        '-i', step1_file,
                        '-vf', f"subtitles='{vtt_path_safe}':force_style='{style}'", 
                        '-c:a', 'copy',
                        out_path
                    ]
                    subprocess.run(cmd, check=True, timeout=300)
                    
                    if os.path.exists(out_path):
                        final_output_path = out_path
                        try: os.remove(step1_file)
                        except: pass
                    else:
                        final_output_path = step1_file
                except Exception as e:
                    print(f"Subtitle burn failed: {e}")
                    if os.path.exists(step1_file):
                        shutil.copy(step1_file, out_path)
                    final_output_path = out_path
            else:
                if os.path.exists(step1_file):
                    shutil.move(step1_file, out_path)
                final_output_path = out_path

            url = f"/download/{os.path.basename(final_output_path)}"
            tasks[task_id] = {
                'percent': 100,
                'status': 'Selesai!',
                'result': {'original': manual_text, 'translated': translated_text, 'url': url, 'type': 'video'}
            }
            return
        except Exception as e:
            print(f"IMAGE ERROR: {e}")
            import traceback; traceback.print_exc()
            tasks[task_id] = {'percent': 0, 'status': 'Error', 'error': str(e)}
            return
        finally:
            cleanup_temp(temp_files)

    # ==========================================
    # COMPRESS MODE
    # ==========================================
    if mode == 'compress':
        try:
            tasks[task_id] = {'percent': 10, 'status': 'Compressing Video...', 'result': None}
            print(f"[{task_id}] Compressing Video (Level: {compress_level})...")
            
            video = mp.VideoFileClip(file_path)
            
            if compress_level == 'whatsapp':
                if video.w > 854: video = video.resize(width=854)
                bitrate = '500k'
            elif compress_level == 'high':
                if video.w > 1280: video = video.resize(width=1280)
                bitrate = '700k'
            elif compress_level == 'medium':
                if video.w > 1920: video = video.resize(width=1920)
                bitrate = '1500k'
            else:
                bitrate = '3000k'

            out_name = f"compressed_{task_id}.mp4"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            video.write_videofile(out_path, codec='libx264', audio_codec='aac', bitrate=bitrate, preset='medium', logger=None)
            video.close()
            
            url = f"/download/{out_name}"
            tasks[task_id] = {
                'percent': 100,
                'status': 'Selesai!',
                'result': {'original': '', 'translated': '', 'url': url, 'type': 'video'}
            }
            return
        except Exception as e:
            print(f"COMPRESS ERROR: {e}")
            import traceback; traceback.print_exc()
            tasks[task_id] = {'percent': 0, 'status': 'Error', 'error': str(e)}
            return
        finally:
            cleanup_temp(temp_files)

    # ==========================================
    # VIDEO DUBBING / SUBTITLE / TTS MODE
    # ==========================================
    try:
        # Fallback: if image file sent to video mode
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.webp']:
            return process_dubbing(task_id, file_path, source_lang, target_lang, video_format, 
                                 gender, enhance_video, unique_mode, manual_text, voice_effect, 
                                 app_host, mode='image', backsound_path=backsound_path, 
                                 animation_mode=animation_mode)

        # ---- STEP 1: PREPARE SOURCE ----
        tasks[task_id] = {'percent': 10, 'status': 'Menyiapkan Sumber...', 'result': None}
        original_text = ""
        audio_path = None
        video_duration = 0
        speech_segments = None       # Will hold timestamped speech segments
        translated_segments = None   # Will hold translated segments with timestamps
        
        if is_tts:
            print(f"[{task_id}] TTS Mode")
            if not manual_text: raise ValueError("TTS wajib pakai Naskah Teks!")
            original_text = manual_text
        elif mode == 'subtitle':
            print(f"[{task_id}] Subtitle Mode")
            # Extract audio for transcription
            tasks[task_id]['status'] = 'Mengekstrak Audio...'
            video_clip = mp.VideoFileClip(file_path)
            video_duration = video_clip.duration
            
            audio_filename = f"{task_id}.wav"
            audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
            temp_files.append(audio_path)
            
            if video_clip.audio:
                video_clip.audio.write_audiofile(audio_path, logger=None)
            video_clip.close()
            
            # Use manual text or transcribe
            if manual_text and len(manual_text) > 5:
                original_text = manual_text
            else:
                if source_lang == 'auto' and audio_path and os.path.exists(audio_path):
                    tasks[task_id]['status'] = 'AI Mendeteksi Bahasa...'
                    source_lang = detect_language_audio(audio_path)
                
                tasks[task_id]['percent'] = 30
                tasks[task_id]['status'] = f'Mendengarkan ({source_lang})...'
                original_text = transcribe_large_audio(audio_path, source_lang)
                if not original_text: original_text = "No speech detected."
        else:
            # VIDEO DUBBING
            print(f"[{task_id}] Video Dubbing Mode")
            
            tasks[task_id]['status'] = 'Mengekstrak Audio...'
            video_clip = mp.VideoFileClip(file_path)
            video_duration = video_clip.duration
            print(f"[{task_id}] Video Duration: {video_duration:.2f}s")
            
            audio_filename = f"{task_id}.wav"
            audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
            temp_files.append(audio_path)
            
            if video_clip.audio:
                video_clip.audio.write_audiofile(audio_path, logger=None)
            else:
                # Silent video - create silent wav
                cmd = f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {video_duration} -acodec pcm_s16le "{audio_path}" -y'
                os.system(cmd)
            video_clip.close()
            
            # Auto-Detect Language
            if source_lang == 'auto':
                tasks[task_id]['status'] = 'AI Mendeteksi Bahasa...'
                source_lang = detect_language_audio(audio_path)
                print(f"[{task_id}] Detected: {source_lang}")

            # Auto-Detect Gender
            if gender == 'auto':
                tasks[task_id]['status'] = 'AI Mendeteksi Gender...'
                gender = detect_gender(audio_path)
                
            # Transcribe with speech-level timestamps
            if manual_text and len(manual_text) > 5:
                original_text = manual_text
                speech_segments = None  # Will use sentence-based sync
            else:
                tasks[task_id]['percent'] = 20
                tasks[task_id]['status'] = 'Mendeteksi Bicara...'
                print(f"[{task_id}] 🔍 Detecting speech segments...")
                
                # Step 1: Find where speech occurs in the video
                speech_times = detect_speech_segments(audio_path)
                
                tasks[task_id]['percent'] = 30
                tasks[task_id]['status'] = f'Mendengarkan {len(speech_times)} Segmen...'
                print(f"[{task_id}] 🎤 Transcribing {len(speech_times)} speech segments...")
                
                # Step 2: Transcribe each speech segment individually
                speech_segments = transcribe_with_timestamps(audio_path, source_lang, speech_times)
                
                # Assemble full text for display
                original_text = " ".join([s['text'] for s in speech_segments])
                if not original_text.strip():
                    original_text = "No speech detected."
                    speech_segments = None
                else:
                    print(f"[{task_id}] ✅ Transcribed {len(speech_segments)} segments: \"{original_text[:80]}...\"")

        # ---- STEP 2: TRANSLATE ----
        # For speech-segment mode, translate per-segment to preserve timestamps
        if speech_segments and source_lang != target_lang:
            tasks[task_id]['percent'] = 50
            tasks[task_id]['status'] = 'Menerjemahkan per-Segmen...'
            print(f"[{task_id}] Translating {len(speech_segments)} segments...")
            translated_segments = translate_segments(speech_segments, source_lang, target_lang)
            translated_text = " ".join([s['text'] for s in translated_segments])
        elif speech_segments and source_lang == target_lang:
            translated_segments = [{**s, 'original': s['text']} for s in speech_segments]
            translated_text = original_text
        else:
            translated_segments = None
            if source_lang == target_lang:
                translated_text = original_text
            else:
                tasks[task_id]['percent'] = 50
                tasks[task_id]['status'] = 'Menerjemahkan...'
                try:
                    src_code = source_lang.split('-')[0] if 'zh' not in source_lang else source_lang
                    tgt_code = target_lang.split('-')[0] if 'zh' not in target_lang else target_lang
                    if tgt_code == 'zh': tgt_code = 'zh-CN'
                    s_code = src_code if src_code else 'auto'
                    trans = GoogleTranslator(source=s_code, target=tgt_code)
                    translated_text = translate_large_text(original_text, trans)
                except: 
                    translated_text = original_text

        # ---- STEP 3: GENERATE SPEECH + SYNC ----
        tasks[task_id]['percent'] = 70
        tasks[task_id]['status'] = f'Membuat Suara AI ({gender})...'
        
        tts_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_tts.mp3")
        temp_files.append(tts_path)
        
        if gender == 'auto': gender = 'female'
        voice = VOICE_MAP.get(target_lang, {}).get(gender, 'en-US-JennyNeural')
        effect_params = EFFECTS_MAP.get(voice_effect, EFFECTS_MAP['normal'])
        base_rate = effect_params['rate']
        base_pitch = effect_params['pitch']
        
        vtt_path = tts_path.replace('.mp3', '.vtt')
        
        if mode == 'subtitle':
            # ---- SUBTITLE MODE: No TTS voice, just create subtitle file ----
            print(f"[{task_id}] Subtitle Mode: Creating subtitles only (no voice)...")
            
            # Create silent audio placeholder
            cmd = f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t 1 -q:a 9 -acodec libmp3lame "{tts_path}" -y'
            os.system(cmd)
            
            # Generate properly synced VTT
            sentences = split_text_to_sentences(translated_text)
            vtt_content = generate_synced_vtt(sentences, video_duration)
            with open(vtt_path, 'w', encoding='utf-8') as f:
                f.write(vtt_content)
            print(f"[{task_id}] Created {len(sentences)} subtitle blocks for {video_duration:.1f}s video")
            
        elif dub_audio_path and os.path.exists(dub_audio_path):
            # ---- UPLOADED DUB AUDIO: Use user's audio file ----
            print(f"[{task_id}] Using Uploaded Dub Audio: {dub_audio_path}")
            shutil.copy(dub_audio_path, tts_path)
            
            # Generate VTT from translated text synced to uploaded audio duration
            try:
                dub_clip = mp.AudioFileClip(tts_path)
                dub_dur = dub_clip.duration
                dub_clip.close()
            except:
                dub_dur = video_duration if video_duration > 0 else 10
            
            sentences = split_text_to_sentences(translated_text)
            vtt_content = generate_synced_vtt(sentences, dub_dur)
            with open(vtt_path, 'w', encoding='utf-8') as f:
                f.write(vtt_content)
                
        else:
            # ---- NORMAL TTS: Generate synced audio + subtitles ----
            if not translated_text or not translated_text.strip():
                translated_text = "..."
            
            # Use synced generation if we have video duration (for video mode)
            if video_duration > 1 and not is_tts:
                
                if translated_segments and sync_mode != 'video_to_audio':
                    # === SPEECH-LEVEL SYNC (BEST QUALITY) ===
                    # Uses detected speech timestamps for accurate dubbing
                    print(f"[{task_id}] 🎯 Speech-Level Sync: {len(translated_segments)} segments...")
                    tasks[task_id]['status'] = f'Dubbing {len(translated_segments)} Segmen Bicara...'
                    
                    # Get voice map for per-segment gender matching
                    lang_voices = VOICE_MAP.get(target_lang, {})
                    
                    combined_path, vtt_content = generate_speech_synced_audio(
                        task_id, translated_segments, voice, video_duration, base_rate, base_pitch,
                        voice_map=lang_voices, target_lang=target_lang
                    )
                    
                    if vtt_content:
                        with open(vtt_path, 'w', encoding='utf-8') as f:
                            f.write(vtt_content)
                    
                    print(f"[{task_id}] ✅ Speech-level sync complete!")
                
                elif sync_mode == 'video_to_audio':
                    # === VIDEO FOLLOWS AUDIO MODE ===
                    # Generate TTS naturally (free duration), video will adjust
                    print(f"[{task_id}] 🎬 Video-Follows-Audio: Generating natural TTS...")
                    tasks[task_id]['status'] = 'Membuat Suara Natural...'
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(generate_tts_edge(translated_text, voice, tts_path, rate=base_rate, pitch=base_pitch))
                    except Exception as e:
                        print(f"[{task_id}] TTS Error: {e}")
                    finally:
                        loop.close()
                    
                    # Read actual TTS duration
                    if os.path.exists(tts_path) and os.path.getsize(tts_path) > 0:
                        tts_clip_check = mp.AudioFileClip(tts_path)
                        tts_natural_dur = tts_clip_check.duration
                        tts_clip_check.close()
                        print(f"[{task_id}] Natural TTS Duration: {tts_natural_dur:.1f}s vs Video: {video_duration:.1f}s")
                    
                    print(f"[{task_id}] ✅ Natural TTS done, video will be adjusted to match")
                    
                else:
                    # === SENTENCE-BASED SYNC (fallback, for manual text) ===
                    sentences = split_text_to_sentences(translated_text)
                    print(f"[{task_id}] 🚀 Sentence-Based Sync ({len(sentences)} sentences, {video_duration:.1f}s video)...")
                    
                    combined_path, vtt_content = generate_synced_audio(
                        task_id, sentences, voice, video_duration, base_rate, base_pitch
                    )
                    
                    if vtt_content:
                        with open(vtt_path, 'w', encoding='utf-8') as f:
                            f.write(vtt_content)
                    
                    print(f"[{task_id}] ✅ Sentence sync complete!")
                
            else:
                # TTS mode or very short video: single-shot generation
                print(f"[{task_id}] Single-shot TTS generation...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(generate_tts_edge(translated_text, voice, tts_path, rate=base_rate, pitch=base_pitch))
                except Exception as e:
                    print(f"[{task_id}] TTS Error: {e}")
                    cmd = f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t 3 -q:a 9 -acodec libmp3lame "{tts_path}" -y'
                    os.system(cmd)
                finally:
                    loop.close()

        # ---- STEP 4: SUBTITLE TRANSLATION (if different from voice language) ----
        if os.path.exists(vtt_path) and subtitle_lang != 'match' and subtitle_lang != target_lang:
            print(f"[{task_id}] Translating Subtitles ({target_lang} -> {subtitle_lang})...")
            try:
                with open(vtt_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                src_code = target_lang.split('-')[0] if 'zh' not in target_lang else target_lang
                tgt_code = subtitle_lang.split('-')[0] if 'zh' not in subtitle_lang else subtitle_lang
                if tgt_code == 'zh': tgt_code = 'zh-CN'
                
                if src_code != tgt_code:
                    trans = GoogleTranslator(source=src_code, target=tgt_code)
                    new_lines = []
                    for line in lines:
                        stripped = line.strip()
                        if "-->" in stripped or stripped == "WEBVTT" or stripped.isdigit() or stripped == "":
                            new_lines.append(line)
                        else:
                            try:
                                tr_line = trans.translate(stripped)
                                new_lines.append(tr_line + "\n")
                            except:
                                new_lines.append(line)
                    
                    with open(vtt_path, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
            except Exception as e:
                print(f"Sub Translation Error: {e}")

        # ---- STEP 5: MERGE & MASTERING ----
        tasks[task_id]['percent'] = 90
        tasks[task_id]['status'] = 'Merender Video Final...'
        
        if not os.path.exists(tts_path):
            raise Exception("TTS file not found")
        
        new_audio = mp.AudioFileClip(tts_path)
        
        if is_tts or video_format == 'audio':
            # ---- AUDIO-ONLY OUTPUT ----
            final_audio = new_audio
            
            # Mix backsound if any
            if backsound_path and os.path.exists(backsound_path):
                try:
                    bg_audio = mp.AudioFileClip(backsound_path)
                    target_dur = final_audio.duration
                    if bg_audio.duration < target_dur:
                        n = int(target_dur / bg_audio.duration) + 2
                        bg_audio = mp.concatenate_audioclips([bg_audio] * n)
                    safe_dur = min(target_dur, bg_audio.duration - 0.01)
                    bg_audio = bg_audio.subclip(0, safe_dur).volumex(0.25)
                    final_audio = CompositeAudioClip([final_audio, bg_audio])
                except Exception as e:
                    print(f"Backsound mix error: {e}")
            
            out_name = f"audio_{task_id}.mp3"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            final_audio.write_audiofile(out_path, logger=None)
            result_type = 'audio'
            
            try: final_audio.close()
            except: pass
            try: new_audio.close()
            except: pass
            
        else:
            # ---- VIDEO OUTPUT ----
            video_clip = mp.VideoFileClip(file_path)
            final_video = video_clip
            vd = video_clip.duration
            
            # Build final audio mix
            final_audio = new_audio
            mixed_audio_clips = []
            
            if mode == 'subtitle':
                # Subtitle mode: use original audio as main
                mixed_audio_clips = []
                if audio_path and os.path.exists(audio_path):
                    try:
                        orig_audio = mp.AudioFileClip(audio_path).volumex(1.0)
                        mixed_audio_clips.append(orig_audio)
                    except Exception as e:
                        print(f"Original audio load error: {e}")
            else:
                # Dubbing mode: TTS is main audio
                mixed_audio_clips = [final_audio]
                
                # Mix original audio as background if requested
                if keep_original == 'true' and audio_path and os.path.exists(audio_path):
                    print(f"[{task_id}] Mixing original audio (SFX/ambient) as background...")
                    try:
                        # Volume 0.5 to preserve SFX, impacts, ambient sounds from original film
                        orig_audio = mp.AudioFileClip(audio_path).volumex(0.5)
                        if orig_audio.duration > final_audio.duration:
                            orig_audio = orig_audio.subclip(0, min(final_audio.duration, orig_audio.duration - 0.01))
                        mixed_audio_clips.append(orig_audio)
                    except Exception as e:
                        print(f"Original audio mix error: {e}")
            
            # Mix backsound
            if backsound_path and os.path.exists(backsound_path):
                print(f"[{task_id}] Mixing Backsound...")
                try:
                    bg_audio = mp.AudioFileClip(backsound_path)
                    target_dur = vd
                    if bg_audio.duration < target_dur:
                        n = int(target_dur / bg_audio.duration) + 2
                        bg_audio = mp.concatenate_audioclips([bg_audio] * n)
                    safe_dur = min(target_dur, bg_audio.duration - 0.01)
                    bg_audio = bg_audio.subclip(0, safe_dur).volumex(0.25)
                    mixed_audio_clips.append(bg_audio)
                except Exception as e:
                    print(f"Backsound Mix Failed: {e}")
            
            # Composite all audio
            if len(mixed_audio_clips) > 1:
                final_audio = CompositeAudioClip(mixed_audio_clips)
            elif len(mixed_audio_clips) == 1:
                final_audio = mixed_audio_clips[0]
            
            # ---- DURATION HANDLING ----
            if sync_mode == 'video_to_audio' and mode != 'subtitle':
                # === VIDEO FOLLOWS AUDIO ===
                # Speed up or slow down video to match audio duration
                audio_dur = final_audio.duration
                if abs(audio_dur - vd) > 0.5:  # Only adjust if difference > 0.5s
                    speed_factor = vd / audio_dur  # > 1 = speed up video, < 1 = slow down
                    # Clamp speed to reasonable range (0.5x - 2.0x)
                    speed_factor = max(0.5, min(2.0, speed_factor))
                    print(f"[{task_id}] 🎬 Video speed adjusted: {speed_factor:.2f}x (Video {vd:.1f}s -> Audio {audio_dur:.1f}s)")
                    tasks[task_id]['status'] = f'Menyesuaikan Video ({speed_factor:.2f}x)...'
                    final_video = final_video.fx(vfx.speedx, speed_factor)
                    vd = final_video.duration  # Update video duration
                
                # After speed adjustment, pad/trim to exactly match audio
                if final_audio.duration > final_video.duration:
                    try:
                        last_frame = final_video.to_ImageClip(t=final_video.duration - 0.1).set_duration(final_audio.duration - final_video.duration)
                        final_video = concatenate_videoclips([final_video, last_frame])
                    except:
                        final_audio = final_audio.subclip(0, min(final_video.duration, final_audio.duration - 0.01))
                
                if final_audio.duration < final_video.duration:
                    final_audio = CompositeAudioClip([final_audio]).set_duration(final_video.duration)
                
                final_video = final_video.set_audio(final_audio)
            else:
                # === AUDIO FOLLOWS VIDEO (default) ===
                # Keep original video duration. Extend or trim audio to match.
                if final_audio.duration > vd:
                    # Audio longer than video: freeze last frame to extend video
                    print(f"[{task_id}] Audio ({final_audio.duration:.1f}s) > Video ({vd:.1f}s). Extending video...")
                    try:
                        last_frame = final_video.to_ImageClip(t=vd - 0.1).set_duration(final_audio.duration - vd)
                        final_video = concatenate_videoclips([final_video, last_frame])
                    except Exception as e:
                        print(f"[{task_id}] Extend video failed: {e}, trimming audio instead")
                        final_audio = final_audio.subclip(0, min(vd, final_audio.duration - 0.01))
                
                if final_audio.duration < final_video.duration:
                    # Audio shorter than video: pad audio with silence
                    final_audio = CompositeAudioClip([final_audio]).set_duration(final_video.duration)
                
                final_video = final_video.set_audio(final_audio)
            
            # Anti-Copyright modifications
            if unique_mode == 'true':
                print(f"[{task_id}] 🛡️ Applying Anti-Copyright...")
                
                # 1. Slight speed change (1.02-1.05x)
                random_speed = random.uniform(1.02, 1.05)
                print(f"   -> Speed: {random_speed:.3f}x")
                final_video = final_video.fx(vfx.speedx, random_speed)
                
                # 2. Slight crop/zoom (2-4% zoom in to change frame)
                w, h = final_video.size
                crop_pct = random.uniform(0.02, 0.04)
                crop_x = int(w * crop_pct / 2)
                crop_y = int(h * crop_pct / 2)
                final_video = final_video.crop(x1=crop_x, y1=crop_y, x2=w-crop_x, y2=h-crop_y)
                final_video = final_video.resize((w, h))  # Resize back to original dimensions
                print(f"   -> Crop/Zoom: {crop_pct*100:.1f}%")
                
                # 3. Random horizontal flip (50% chance)
                if random.random() > 0.5:
                    final_video = final_video.fx(vfx.mirror_x)
                    print(f"   -> Mirror: Horizontal")
                
                # 4. Color/brightness variation
                color_factor = random.uniform(1.02, 1.08)
                final_video = final_video.fx(vfx.colorx, color_factor)
                print(f"   -> Color: {color_factor:.2f}x")
                
                # 5. Subtle brightness shift via gamma
                gamma = random.uniform(0.95, 1.05)
                final_video = final_video.fx(vfx.gamma_corr, gamma)
                print(f"   -> Gamma: {gamma:.2f}")
                
                print(f"[{task_id}] ✅ Anti-Copyright applied!")

            if enhance_video == 'true': 
                final_video = final_video.fx(vfx.colorx, 1.1)

            # Resizing
            if video_format == '1080p':
                final_video = final_video.resize(height=1080)
                w, h = final_video.size
                if w % 2 != 0:
                    final_video = final_video.crop(x1=0, y1=0, width=w-1, height=h)
            elif video_format == 'landscape': 
                final_video = final_video.resize(height=720)
                w, h = final_video.size
                if w % 2 != 0:
                    final_video = final_video.crop(x1=0, y1=0, width=w-1, height=h)
            elif video_format == 'portrait':
                w, h = final_video.size
                if w > h:
                    new_w = h * (9/16)
                    if int(new_w) % 2 != 0: new_w = int(new_w) - 1
                    else: new_w = int(new_w)
                    x1 = (w/2) - (new_w/2)
                    final_video = final_video.crop(x1=x1, y1=0, width=new_w, height=h)

            out_name = f"dubbed_{task_id}.mp4"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            
            # Write video (step 1: without subtitles)
            step1_file = os.path.join(UPLOAD_FOLDER, f"{task_id}_step1.mp4")
            print(f"[{task_id}] Writing video to {step1_file}...")
            final_video.write_videofile(step1_file, codec='libx264', audio_codec='aac', 
                                       audio_bitrate='192k', fps=24, threads=1, preset='ultrafast')
            
            try: final_audio.close()
            except: pass
            try: video_clip.close()
            except: pass
            
            # ---- STEP 6: BURN SUBTITLES ----
            has_subs = os.path.exists(vtt_path) and os.path.getsize(vtt_path) > 20
            final_output_path = step1_file
            
            if has_subs and burn_subtitles == 'true':
                print(f"[{task_id}] Burning subtitles into video...")
                tasks[task_id]['status'] = 'Membakar Subtitle...'
                try:
                    vtt_path_safe = vtt_path.replace('\\', '/')
                    
                    try:
                        import imageio_ffmpeg
                        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    except:
                        ffmpeg_exe = 'ffmpeg'
                    
                    style = "Fontname=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000,BorderStyle=4,Outline=1,Shadow=0,Alignment=2,MarginV=25"
                    cmd = [
                        ffmpeg_exe, '-y', 
                        '-i', step1_file,
                        '-vf', f"subtitles='{vtt_path_safe}':force_style='{style}'", 
                        '-c:a', 'copy',
                        out_path
                    ]
                    # Timeout scales with video duration (min 10 min, +2 min per minute of video)
                    ffmpeg_timeout = max(600, int(vd * 2) + 600)
                    subprocess.run(cmd, check=True, timeout=ffmpeg_timeout)
                    
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                        final_output_path = out_path
                        try: os.remove(step1_file)
                        except: pass
                    else:
                        print(f"[{task_id}] Subtitle output is empty, using original")
                        final_output_path = step1_file
                except Exception as e:
                    print(f"[{task_id}] Burn Subtitles Failed: {e}")
                    if os.path.exists(step1_file):
                        shutil.copy(step1_file, out_path)
                    final_output_path = out_path
            else:
                if os.path.exists(step1_file):
                    shutil.move(step1_file, out_path)
                final_output_path = out_path
            
            result_type = 'video'

        # ---- EXPOSE VTT FOR DOWNLOAD ----
        vtt_url = None
        if os.path.exists(vtt_path):
            vtt_out_name = f"sub_{task_id}.vtt"
            vtt_out_path = os.path.join(OUTPUT_FOLDER, vtt_out_name)
            shutil.copy(vtt_path, vtt_out_path)
            vtt_url = f"/download/{vtt_out_name}"

        url = f"/download/{out_name}"
        tasks[task_id] = {
            'percent': 100,
            'status': 'Selesai!',
            'result': {'original': original_text, 'translated': translated_text, 'url': url, 'vtt_url': vtt_url, 'type': result_type}
        }
        print(f"[{task_id}] ✅ DONE! Output: {out_name}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        tasks[task_id] = {'percent': 0, 'status': 'Error', 'error': str(e)}
    finally:
        cleanup_temp(temp_files)


# ==========================================
# LOGIN PAGE HTML
# ==========================================
LOGIN_HTML = '''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - AI Dubbing Studio Pro</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        }
        .container {
            display: flex;
            gap: 30px;
            max-width: 850px;
            width: 95vw;
            flex-wrap: wrap;
            justify-content: center;
        }
        .card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 40px 30px;
            width: 380px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            transition: all 0.3s;
        }
        .card:hover { transform: translateY(-5px); }
        .card.pro { border-color: rgba(168,85,247,0.4); }
        .badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 15px;
            letter-spacing: 1px;
        }
        .badge-free { background: rgba(34,197,94,0.2); color: #4ade80; }
        .badge-pro { background: rgba(168,85,247,0.2); color: #c084fc; }
        .logo { font-size: 40px; margin-bottom: 10px; }
        h2 { color: #fff; font-size: 22px; margin-bottom: 8px; }
        .price { color: rgba(255,255,255,0.5); font-size: 14px; margin-bottom: 20px; }
        .price b { color: #fff; font-size: 20px; }
        .features {
            text-align: left;
            margin-bottom: 25px;
        }
        .features li {
            color: rgba(255,255,255,0.7);
            font-size: 13px;
            list-style: none;
            padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .features li::before { margin-right: 8px; }
        .features li.yes::before { content: "\\2705"; }
        .features li.no::before { content: "\\274C"; }
        .features li.no { color: rgba(255,255,255,0.3); }
        .input-group {
            margin-bottom: 15px;
            text-align: left;
        }
        .input-group label {
            color: rgba(255,255,255,0.6);
            font-size: 12px;
            display: block;
            margin-bottom: 5px;
        }
        .input-group input {
            width: 100%;
            padding: 12px 14px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px;
            color: #fff;
            font-size: 14px;
            outline: none;
            transition: all 0.3s;
        }
        .input-group input:focus {
            border-color: #7c3aed;
            background: rgba(124,58,237,0.1);
        }
        .btn {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 12px;
            color: #fff;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn-demo {
            background: linear-gradient(135deg, #059669, #10b981);
        }
        .btn-demo:hover {
            box-shadow: 0 8px 25px rgba(16,185,129,0.4);
        }
        .btn-pro {
            background: linear-gradient(135deg, #7c3aed, #a855f7);
        }
        .btn-pro:hover {
            box-shadow: 0 8px 25px rgba(124,58,237,0.4);
        }
        .error {
            background: rgba(239,68,68,0.15);
            border: 1px solid rgba(239,68,68,0.3);
            color: #f87171;
            padding: 8px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 13px;
        }
        .contact {
            color: rgba(255,255,255,0.4);
            font-size: 12px;
            margin-top: 15px;
        }
        .contact a { color: #a855f7; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <!-- DEMO CARD -->
        <div class="card">
            <span class="badge badge-free">GRATIS</span>
            <div class="logo">&#127911;</div>
            <h2>Demo</h2>
            <div class="price"><b>Rp 0</b> / selamanya</div>
            <ul class="features">
                <li class="yes">Video dubbing</li>
                <li class="yes">Subtitle otomatis</li>
                <li class="yes">Multi bahasa</li>
                <li class="no">Maks 30 detik video</li>
                <li class="no">Anti-Copyright</li>
                <li class="no">Mode Sync Audio</li>
                <li class="no">Keep Original SFX</li>
            </ul>
            <form method="POST" action="/login">
                <input type="hidden" name="demo" value="true">
                <button type="submit" class="btn btn-demo">Coba Demo Gratis</button>
            </form>
        </div>

        <!-- PRO CARD -->
        <div class="card pro">
            <span class="badge badge-pro">PRO</span>
            <div class="logo">&#127908;</div>
            <h2>Pro</h2>
            <div class="price"><b>Premium</b> / akses penuh</div>
            <ul class="features">
                <li class="yes">Video dubbing tanpa batas</li>
                <li class="yes">Subtitle otomatis</li>
                <li class="yes">Multi bahasa</li>
                <li class="yes">Video panjang unlimited</li>
                <li class="yes">Anti-Copyright</li>
                <li class="yes">Mode Sync Audio</li>
                <li class="yes">Keep Original SFX</li>
            </ul>
            {{ error_html }}
            <form method="POST" action="/login">
                <div class="input-group">
                    <label>Username</label>
                    <input type="text" name="username" placeholder="Username Pro">
                </div>
                <div class="input-group">
                    <label>Password</label>
                    <input type="password" name="password" placeholder="Password">
                </div>
                <button type="submit" class="btn btn-pro">Login Pro</button>
            </form>
            <p class="contact">Hubungi admin untuk akun Pro</p>
        </div>
    </div>
</body>
</html>
'''

# ==========================================
# FLASK ROUTES
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Demo login (no credentials needed)
        if request.form.get('demo') == 'true':
            session['logged_in'] = True
            session['username'] = 'demo'
            session['role'] = 'demo'
            return redirect('/')
        
        # Pro login
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username in USERS and USERS[username]['password'] == password:
            session['logged_in'] = True
            session['username'] = username
            session['role'] = USERS[username]['role']
            return redirect('/')
        else:
            error = '<div class="error">Username atau password salah!</div>'
            return render_template_string(LOGIN_HTML, error_html=error)
    return render_template_string(LOGIN_HTML, error_html='')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/user-info')
def user_info():
    if not session.get('logged_in'):
        return jsonify({'logged_in': False})
    return jsonify({
        'logged_in': True,
        'username': session.get('username', ''),
        'role': session.get('role', 'demo')
    })

@app.route('/')
def home():
    if not session.get('logged_in'):
        return redirect('/login')
    if os.path.exists('dub.html'): return send_file('dub.html')
    if os.path.exists('voice.html'): return send_file('voice.html')
    return "Not Found"

@app.route('/dub-video', methods=['POST'])
def start_job():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized. Please login.'}), 401
    
    # Demo user restrictions
    user_role = session.get('role', 'demo')
    
    try:
        mode = request.form.get('mode', 'video')
        
        if mode in ['video', 'image', 'compress', 'subtitle']:
            if 'video' not in request.files: return jsonify({'error': 'No file'}), 400
            file = request.files['video']
            if not file.filename: return jsonify({'error': 'No file'}), 400
            filename = file.filename
        else:
            filename = "tts_request.txt" 
            file = None

        source_lang = request.form.get('source_lang', 'auto') 
        target_lang = request.form.get('target_lang', 'id-ID')
        video_format = request.form.get('video_format', 'original')
        gender = request.form.get('gender', 'auto') 
        enhance_video = request.form.get('enhance_video', 'false')
        unique_mode = request.form.get('unique_mode', 'false')
        manual_text = request.form.get('source_text', '') 
        voice_effect = request.form.get('voice_effect', 'normal') 
        animation_mode = request.form.get('animation', 'cinematic')
        compress_lvl = request.form.get('compress_level', 'medium')
        subtitle_lang = request.form.get('subtitle_lang', 'match')
        sync_video = request.form.get('sync_video_to_audio', 'false')
        sync_mode = 'video_to_audio' if sync_video == 'true' else 'audio_to_video'
        
        # Demo user restrictions - disable pro features
        if user_role == 'demo':
            unique_mode = 'false'  # No anti-copyright
            sync_video = 'false'   # No sync video
            sync_mode = 'audio_to_video'
        
        if mode == 'subtitle':
            keep_orig = 'true'
            burn_subs = 'true'
        else:
            keep_orig = request.form.get('keep_original_audio', 'false')
            burn_subs = request.form.get('burn_subtitles', 'false')
        
        tid = str(uuid.uuid4())
        sp = ""
        if file:
            sp = os.path.join(UPLOAD_FOLDER, f"{tid}_{filename}")
            file.save(sp)
            
        bs_path = None
        if 'backsound' in request.files:
            bs_file = request.files['backsound']
            if bs_file.filename:
                bs_path = os.path.join(UPLOAD_FOLDER, f"{tid}_bg_{bs_file.filename}")
                bs_file.save(bs_path)

        dub_path = None
        if 'dub_audio' in request.files:
            dub_file = request.files['dub_audio']
            if dub_file.filename:
                dub_path = os.path.join(UPLOAD_FOLDER, f"{tid}_dub_{dub_file.filename}")
                dub_file.save(dub_path)
        
        app_host = request.host.split(':')[0]
        t = threading.Thread(target=process_dubbing, args=(
            tid, sp, source_lang, target_lang, video_format, gender, 
            enhance_video, unique_mode, manual_text, voice_effect, app_host, 
            mode, bs_path, animation_mode, compress_lvl, keep_orig, burn_subs, 
            subtitle_lang, dub_path, sync_mode
        ))
        t.start()
        return jsonify({'task_id': tid, 'status': 'queued'})
    except Exception as e: 
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<task_id>')
def get_progress(task_id):
    return jsonify(tasks.get(task_id, {'error': 'Not Found'}))

@app.route('/view/<path:filename>')
def view_file_route(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=False)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

def start_ngrok():
    from pyngrok import ngrok, conf
    import time
    
    token = os.environ.get("NGROK_AUTHTOKEN", "39WcL26evTX905KDjdWgS5v6OQ5_6JHqe4y856PthNbbchtBt")
    
    if token:
        print(f"Setting Ngrok Token: {token[:4]}...")
        ngrok.set_auth_token(token)
    
    print(" * Cleaning up old ngrok processes...")
    try:
        ngrok.kill()
    except: pass
    os.system("taskkill /f /im ngrok.exe >nul 2>&1")
    time.sleep(2)
    
    public_url = None
    
    try:
        print(" * Connecting to Ngrok (Preferred Domain)...")
        public_url = ngrok.connect(5000, "http", domain="semitheatric-konnor-superadjacent.ngrok-free.dev").public_url
    except Exception as e:
        print(f"⚠️ Preferred domain failed: {e}")
        print(" * Fallback: Starting Random Domain Tunnel...")
        try:
            os.system("taskkill /f /im ngrok.exe >nul 2>&1")
            time.sleep(1)
            public_url = ngrok.connect(5000, "http").public_url
        except Exception as e2:
            print(f"❌ Ngrok Fatal Error: {e2}")
            return None

    if public_url:
        print(f" * Public URL: {public_url}")
    
    return public_url

if __name__ == '__main__':
    if '--offline' not in os.sys.argv:
        print("Starting Online Mode (Ngrok)...")
        t = threading.Thread(target=start_ngrok)
        t.start()
    else:
        print("Offline Mode (Local only)")
        
    app.run(host='0.0.0.0', port=5000, debug=True)
