"""
AI-Powered Physiotherapist with Natural Language Processing
Provides intelligent, contextual feedback like a real physiotherapist
"""

import numpy as np
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import random
import pyttsx3
from gtts import gTTS
import pygame
import tempfile
import os

# AI and NLP imports for intelligent feedback
try:
    from transformers import pipeline
    USE_AI = True
except ImportError:
    USE_AI = False
    print("Warning: Transformers not installed. Using rule-based feedback.")

@dataclass
class MovementPattern:
    """Represents a movement pattern for analysis"""
    exercise: str
    angles: Dict[str, float]
    velocity: float
    acceleration: float
    form_score: float
    timestamp: float

@dataclass
class PhysiotherapistFeedback:
    """Structured feedback from AI physiotherapist"""
    text: str
    priority: int  # 1-5, 5 being highest
    category: str  # 'form', 'motivation', 'instruction', 'warning', 'praise'
    emotion: str   # 'neutral', 'encouraging', 'concerned', 'excited'

class AIPhysiotherapist:
    """Advanced AI-powered physiotherapist for real-time coaching"""
    
    def __init__(self):
        """Initialize the AI physiotherapist system"""
        # Voice configuration
        self.voice_engine = None
        self.use_gtts = True  # Use Google TTS for better quality
        self._init_voice_system()
        
        # AI/NLP components
        self.sentiment_analyzer = None
        self.text_generator = None
        if USE_AI:
            self._init_ai_models()
        
        # Movement analysis
        self.movement_history = deque(maxlen=100)
        self.pattern_buffer = deque(maxlen=30)
        self.feedback_history = deque(maxlen=20)
        
        # Physiotherapist personality
        self.personality = {
            'encouraging': 0.7,
            'technical': 0.5,
            'strict': 0.3,
            'humorous': 0.2
        }
        
        # Feedback management
        self.feedback_queue = queue.PriorityQueue()
        self.last_feedback_time = {}
        self.feedback_cooldowns = {
            'form': 2.0,
            'motivation': 5.0,
            'instruction': 3.0,
            'warning': 1.5,
            'praise': 4.0
        }
        
        # Exercise-specific knowledge base
        self.exercise_knowledge = self._load_exercise_knowledge()
        
        # Performance tracking
        self.session_stats = {
            'total_reps': 0,
            'perfect_form_reps': 0,
            'common_errors': [],
            'improvement_areas': [],
            'session_duration': 0
        }
        
        # Start voice processing thread
        self.voice_thread = threading.Thread(target=self._voice_processor, daemon=True)
        self.voice_thread.start()
        
    def _init_voice_system(self):
        """Initialize voice output system with fallback options"""
        try:
            if self.use_gtts:
                pygame.mixer.init()
                self.voice_engine = 'gtts'
            else:
                self.voice_engine = pyttsx3.init()
                self.voice_engine.setProperty('rate', 170)
                self.voice_engine.setProperty('volume', 0.9)
                
                # Try to set a natural voice
                voices = self.voice_engine.getProperty('voices')
                if voices:
                    # Prefer female voice for physiotherapist
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.voice_engine.setProperty('voice', voice.id)
                            break
        except Exception as e:
            print(f"Voice system initialization error: {e}")
            self.voice_engine = None
    
    def _init_ai_models(self):
        """Initialize AI models for intelligent feedback"""
        try:
            # Use smaller models for real-time performance
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            # For more advanced feedback, could use a conversational model
            # self.text_generator = pipeline("text-generation", model="gpt2")
        except Exception as e:
            print(f"AI model initialization error: {e}")
            self.sentiment_analyzer = None
    
    def _load_exercise_knowledge(self) -> Dict:
        """Load exercise-specific knowledge base"""
        return {
            'squat': {
                'perfect_angles': {'knee': 90, 'hip': 90},
                'angle_tolerance': 15,
                'common_errors': [
                    'knees caving inward',
                    'heels lifting off ground',
                    'back rounding',
                    'not going deep enough',
                    'leaning too far forward'
                ],
                'cues': [
                    "Keep your chest up and proud",
                    "Push through your heels",
                    "Knees should track over your toes",
                    "Engage your core throughout",
                    "Control the descent, don't just drop",
                    "Breathe in on the way down, out on the way up"
                ],
                'motivational': [
                    "Your form is improving with each rep!",
                    "Great depth on that squat!",
                    "You're building serious leg strength!",
                    "Perfect tempo, keep it controlled!",
                    "Your consistency is paying off!"
                ]
            },
            'bicep curl': {
                'perfect_angles': {'elbow': 40, 'shoulder': 0},
                'angle_tolerance': 10,
                'common_errors': [
                    'swinging the weight',
                    'using momentum',
                    'elbows drifting forward',
                    'incomplete range of motion',
                    'rushing the movement'
                ],
                'cues': [
                    "Keep your elbows pinned to your sides",
                    "Control both the lifting and lowering phase",
                    "Squeeze at the top of the movement",
                    "Don't swing your body",
                    "Full extension at the bottom",
                    "Focus on the mind-muscle connection"
                ],
                'motivational': [
                    "Those biceps are working hard!",
                    "Excellent control on that rep!",
                    "You're getting stronger with each curl!",
                    "Perfect isolation of the biceps!",
                    "Great focus and concentration!"
                ]
            },
            'shoulder press': {
                'perfect_angles': {'elbow': 90, 'shoulder': 180},
                'angle_tolerance': 10,
                'common_errors': [
                    'arching the back',
                    'flaring elbows too wide',
                    'incomplete lockout',
                    'pressing forward instead of up',
                    'uneven arm movement'
                ],
                'cues': [
                    "Press straight up, not forward",
                    "Keep your core engaged",
                    "Don't arch your back",
                    "Control the weight on the way down",
                    "Full lockout at the top",
                    "Keep your wrists straight"
                ],
                'motivational': [
                    "Strong press! Your shoulders are getting powerful!",
                    "Excellent stability throughout the movement!",
                    "You're making those weights look easy!",
                    "Perfect form on that press!",
                    "Your shoulder strength is impressive!"
                ]
            }
        }
    
    def analyze_movement(self, exercise: str, angles: Dict[str, float], 
                         landmarks: List, rep_count: int) -> PhysiotherapistFeedback:
        """
        Analyze movement and generate intelligent physiotherapist feedback
        
        Args:
            exercise: Current exercise being performed
            angles: Dictionary of calculated angles
            landmarks: Body landmarks from pose detection
            rep_count: Current repetition count
            
        Returns:
            PhysiotherapistFeedback object with contextual guidance
        """
        # Create movement pattern
        pattern = MovementPattern(
            exercise=exercise,
            angles=angles,
            velocity=self._calculate_velocity(),
            acceleration=self._calculate_acceleration(),
            form_score=self._calculate_form_score(exercise, angles),
            timestamp=time.time()
        )
        
        self.movement_history.append(pattern)
        
        # Generate multi-layered feedback
        feedback = self._generate_intelligent_feedback(pattern, rep_count)
        
        return feedback
    
    def _calculate_velocity(self) -> float:
        """Calculate movement velocity from recent patterns"""
        if len(self.movement_history) < 2:
            return 0.0
        
        recent = list(self.movement_history)[-2:]
        time_diff = recent[1].timestamp - recent[0].timestamp
        
        if time_diff == 0:
            return 0.0
        
        # Calculate angular velocity
        angle_changes = []
        for key in recent[0].angles:
            if key in recent[1].angles:
                change = abs(recent[1].angles[key] - recent[0].angles[key])
                angle_changes.append(change)
        
        if angle_changes:
            return sum(angle_changes) / (len(angle_changes) * time_diff)
        return 0.0
    
    def _calculate_acceleration(self) -> float:
        """Calculate movement acceleration"""
        if len(self.movement_history) < 3:
            return 0.0
        
        recent = list(self.movement_history)[-3:]
        velocities = []
        
        for i in range(1, len(recent)):
            time_diff = recent[i].timestamp - recent[i-1].timestamp
            if time_diff > 0:
                angle_changes = []
                for key in recent[i-1].angles:
                    if key in recent[i].angles:
                        change = abs(recent[i].angles[key] - recent[i-1].angles[key])
                        angle_changes.append(change)
                if angle_changes:
                    velocity = sum(angle_changes) / (len(angle_changes) * time_diff)
                    velocities.append(velocity)
        
        if len(velocities) >= 2:
            return velocities[-1] - velocities[-2]
        return 0.0
    
    def _calculate_form_score(self, exercise: str, angles: Dict[str, float]) -> float:
        """Calculate form score based on angle deviations from ideal"""
        if exercise not in self.exercise_knowledge:
            return 0.5
        
        knowledge = self.exercise_knowledge[exercise]
        perfect_angles = knowledge['perfect_angles']
        tolerance = knowledge['angle_tolerance']
        
        scores = []
        for joint, perfect_angle in perfect_angles.items():
            if joint in angles:
                deviation = abs(angles[joint] - perfect_angle)
                if deviation <= tolerance:
                    score = 1.0 - (deviation / tolerance) * 0.5
                else:
                    score = 0.5 * max(0, 1 - (deviation - tolerance) / tolerance)
                scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _generate_intelligent_feedback(self, pattern: MovementPattern, 
                                      rep_count: int) -> PhysiotherapistFeedback:
        """Generate context-aware physiotherapist feedback"""
        exercise = pattern.exercise
        form_score = pattern.form_score
        
        # Determine feedback type based on context
        if form_score < 0.4:
            return self._generate_correction_feedback(pattern)
        elif form_score > 0.8 and rep_count > 0 and rep_count % 5 == 0:
            return self._generate_praise_feedback(pattern, rep_count)
        elif pattern.velocity > 100:  # Too fast
            return self._generate_tempo_feedback(pattern)
        elif rep_count > 0 and rep_count % 10 == 0:
            return self._generate_motivational_feedback(pattern, rep_count)
        else:
            return self._generate_instructional_feedback(pattern)
    
    def _generate_correction_feedback(self, pattern: MovementPattern) -> PhysiotherapistFeedback:
        """Generate form correction feedback"""
        exercise = pattern.exercise
        knowledge = self.exercise_knowledge.get(exercise, {})
        
        # Identify the biggest form issue
        perfect_angles = knowledge.get('perfect_angles', {})
        worst_deviation = 0
        worst_joint = None
        
        for joint, perfect in perfect_angles.items():
            if joint in pattern.angles:
                deviation = abs(pattern.angles[joint] - perfect)
                if deviation > worst_deviation:
                    worst_deviation = deviation
                    worst_joint = joint
        
        # Generate specific correction
        corrections = {
            'knee': [
                "Focus on your knee angle. Aim for about 90 degrees at the bottom.",
                "Your knees need adjustment. Think about sitting back into a chair.",
                "Watch your knee position. They should track over your toes."
            ],
            'hip': [
                "Adjust your hip position. Keep your chest up and hips back.",
                "Your hip hinge needs work. Push your hips back more.",
                "Focus on your hip mobility. Go deeper if you can."
            ],
            'elbow': [
                "Control your elbow movement. Keep them stable at your sides.",
                "Your elbow angle is off. Focus on the full range of motion.",
                "Watch your elbows. Don't let them drift forward."
            ],
            'shoulder': [
                "Stabilize your shoulders. Keep them down and back.",
                "Your shoulder position needs adjustment. Don't shrug.",
                "Focus on shoulder stability. Press straight up, not forward."
            ]
        }
        
        if worst_joint and worst_joint in corrections:
            text = random.choice(corrections[worst_joint])
        else:
            text = random.choice(knowledge.get('cues', ["Focus on your form. Control the movement."]))
        
        return PhysiotherapistFeedback(
            text=text,
            priority=4,
            category='warning',
            emotion='concerned'
        )
    
    def _generate_praise_feedback(self, pattern: MovementPattern, 
                                 rep_count: int) -> PhysiotherapistFeedback:
        """Generate praise for good form"""
        exercise = pattern.exercise
        knowledge = self.exercise_knowledge.get(exercise, {})
        
        praise_options = [
            f"Excellent form! You've completed {rep_count} perfect reps!",
            f"Outstanding work! {rep_count} reps with textbook technique!",
            f"You're crushing it! {rep_count} reps and your form is spot on!",
            f"Beautiful execution! That's {rep_count} quality repetitions!",
            f"Phenomenal performance! {rep_count} reps of pure perfection!"
        ]
        
        # Add exercise-specific praise
        specific_praise = knowledge.get('motivational', [])
        if specific_praise:
            praise_options.extend(specific_praise)
        
        return PhysiotherapistFeedback(
            text=random.choice(praise_options),
            priority=3,
            category='praise',
            emotion='excited'
        )
    
    def _generate_tempo_feedback(self, pattern: MovementPattern) -> PhysiotherapistFeedback:
        """Generate feedback about movement tempo"""
        tempo_feedback = [
            "Slow down a bit. Focus on control, not speed.",
            "Take your time. Quality over quantity always wins.",
            "Control the tempo. Think 2 seconds down, 2 seconds up.",
            "You're moving too fast. Each phase should be deliberate.",
            "Reduce your speed. Feel the muscle working throughout."
        ]
        
        return PhysiotherapistFeedback(
            text=random.choice(tempo_feedback),
            priority=3,
            category='instruction',
            emotion='neutral'
        )
    
    def _generate_motivational_feedback(self, pattern: MovementPattern, 
                                       rep_count: int) -> PhysiotherapistFeedback:
        """Generate motivational feedback"""
        motivational = [
            f"You're doing great! {rep_count} reps completed. Keep pushing!",
            f"Fantastic effort! {rep_count} down, you've got this!",
            f"You're on fire! {rep_count} reps and counting!",
            f"Incredible work! {rep_count} reps of pure determination!",
            f"You're unstoppable! {rep_count} reps and still going strong!"
        ]
        
        return PhysiotherapistFeedback(
            text=random.choice(motivational),
            priority=2,
            category='motivation',
            emotion='encouraging'
        )
    
    def _generate_instructional_feedback(self, pattern: MovementPattern) -> PhysiotherapistFeedback:
        """Generate general instructional feedback"""
        exercise = pattern.exercise
        knowledge = self.exercise_knowledge.get(exercise, {})
        cues = knowledge.get('cues', [])
        
        if cues:
            text = random.choice(cues)
        else:
            text = "Keep up the good work. Focus on your breathing."
        
        return PhysiotherapistFeedback(
            text=text,
            priority=1,
            category='instruction',
            emotion='neutral'
        )
    
    def speak(self, feedback: PhysiotherapistFeedback, force: bool = False):
        """
        Convert feedback to speech
        
        Args:
            feedback: PhysiotherapistFeedback object
            force: Force immediate speech (bypass cooldown)
        """
        current_time = time.time()
        
        # Check cooldown unless forced
        if not force:
            last_time = self.last_feedback_time.get(feedback.category, 0)
            cooldown = self.feedback_cooldowns.get(feedback.category, 2.0)
            
            if current_time - last_time < cooldown:
                return
        
        self.last_feedback_time[feedback.category] = current_time
        
        # Add to queue with priority
        priority_value = 5 - feedback.priority  # Invert for priority queue
        self.feedback_queue.put((priority_value, current_time, feedback))
    
    def _voice_processor(self):
        """Background thread for processing voice feedback"""
        while True:
            try:
                if not self.feedback_queue.empty():
                    _, _, feedback = self.feedback_queue.get(timeout=0.1)
                    self._speak_text(feedback.text, feedback.emotion)
                else:
                    time.sleep(0.1)
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Voice processor error: {e}")
                time.sleep(0.5)
    
    def _speak_text(self, text: str, emotion: str = 'neutral'):
        """Actually speak the text using TTS"""
        if not self.voice_engine:
            print(f"[Physiotherapist]: {text}")
            return
        
        try:
            if self.voice_engine == 'gtts':
                # Use Google TTS for better quality
                tts = gTTS(text=text, lang='en', slow=False)
                
                # Adjust speech parameters based on emotion
                if emotion == 'excited':
                    tts = gTTS(text=text, lang='en', slow=False, lang_check=False)
                elif emotion == 'concerned':
                    tts = gTTS(text=text, lang='en', slow=True)
                
                # Save to temporary file and play
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tts.save(tmp_file.name)
                    pygame.mixer.music.load(tmp_file.name)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to complete
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    # Clean up
                    pygame.mixer.music.unload()
                    os.unlink(tmp_file.name)
            else:
                # Use pyttsx3 as fallback
                self.voice_engine.say(text)
                self.voice_engine.runAndWait()
                
        except Exception as e:
            print(f"Speech error: {e}")
            print(f"[Physiotherapist]: {text}")
    
    def get_session_summary(self) -> str:
        """Generate a comprehensive session summary"""
        stats = self.session_stats
        
        # Calculate performance metrics
        form_accuracy = (stats['perfect_form_reps'] / max(stats['total_reps'], 1)) * 100
        
        summary = f"""
        Great workout! Here's your session summary:
        
        Total Repetitions: {stats['total_reps']}
        Perfect Form Reps: {stats['perfect_form_reps']}
        Form Accuracy: {form_accuracy:.1f}%
        
        Areas of Improvement:
        {', '.join(stats['improvement_areas']) if stats['improvement_areas'] else 'Keep up the great work!'}
        
        Remember to stretch and stay hydrated. See you next session!
        """
        
        return summary.strip()
    
    def shutdown(self):
        """Clean shutdown of the AI physiotherapist"""
        try:
            if self.voice_engine and self.voice_engine != 'gtts':
                self.voice_engine.stop()
            if self.voice_engine == 'gtts':
                pygame.mixer.quit()
        except:
            pass

class PhysiotherapistPersonality:
    """Different personality profiles for the AI physiotherapist"""
    
    PROFILES = {
        'encouraging': {
            'encouragement': 0.8,
            'strictness': 0.2,
            'technical': 0.3,
            'humor': 0.4
        },
        'strict': {
            'encouragement': 0.3,
            'strictness': 0.8,
            'technical': 0.6,
            'humor': 0.1
        },
        'technical': {
            'encouragement': 0.4,
            'strictness': 0.5,
            'technical': 0.9,
            'humor': 0.2
        },
        'friendly': {
            'encouragement': 0.7,
            'strictness': 0.2,
            'technical': 0.3,
            'humor': 0.7
        }
    }
    
    @staticmethod
    def get_profile(profile_name: str = 'encouraging') -> Dict[str, float]:
        """Get a personality profile"""
        return PhysiotherapistPersonality.PROFILES.get(
            profile_name, 
            PhysiotherapistPersonality.PROFILES['encouraging']
        )
