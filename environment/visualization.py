"""
Malami - Professional Visualization System
==========================================
High-quality 2D visualization for the adaptive learning platform.
"""

import pygame
import numpy as np
from typing import Dict, Any, Optional


class MalamiVisualizer:
    """Professional-grade visualization using Pygame"""
    
    def __init__(self, width: int = 1200, height: int = 700):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Malami - AI Adaptive Learning Platform")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # Colors (modern design system)
        self.colors = {
            "bg_dark": (10, 14, 26),
            "bg_card": (18, 24, 44),
            "primary": (59, 130, 246),
            "success": (16, 185, 129),
            "warning": (245, 158, 11),
            "danger": (239, 68, 68),
            "purple": (139, 92, 246),
            "text": (255, 255, 255),
            "text_muted": (156, 163, 175),
            "border": (45, 58, 94)
        }
        
        self.running = True
        
    def render(self, state: Dict[str, Any], last_action: Optional[Dict] = None):
        """Render the complete learning dashboard"""
        self.screen.fill(self.colors["bg_dark"])
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        
        # Draw header
        self._draw_header()
        
        # Draw main content panels
        self._draw_student_profile(state)
        self._draw_mastery_bars(state)
        self._draw_learning_progress(state)
        self._draw_action_feedback(last_action, state)
        self._draw_recommendations(state)
        
        pygame.display.flip()
        self.clock.tick(30)
    
    def _draw_header(self):
        """Draw application header"""
        header_rect = pygame.Rect(0, 0, self.width, 80)
        pygame.draw.rect(self.screen, self.colors["bg_card"], header_rect)
        pygame.draw.line(self.screen, self.colors["border"], (0, 80), (self.width, 80), 2)
        
        # Title
        title = self.font_large.render("Malami Adaptive Learning Platform", True, self.colors["text"])
        self.screen.blit(title, (30, 25))
        
        # Subtitle
        subtitle = self.font_small.render("AI-Powered Personalized Education", True, self.colors["text_muted"])
        self.screen.blit(subtitle, (30, 55))
    
    def _draw_student_profile(self, state: Dict[str, Any]):
        """Draw student profile card"""
        card = pygame.Rect(30, 100, 350, 250)
        pygame.draw.rect(self.screen, self.colors["bg_card"], card)
        pygame.draw.rect(self.screen, self.colors["border"], card, 2)
        
        # Title
        title = self.font_medium.render("Student Profile", True, self.colors["primary"])
        self.screen.blit(title, (50, 115))
        
        # Metrics
        student_id = state.get('student_id', 'demo_001')
        grade_level = state.get('grade_level', 6)
        learning_rate = state.get('learning_rate', 0.6)
        engagement = state.get('engagement', 0.7)
        fatigue = state.get('fatigue_level', 0.3)
        motivation = state.get('motivation', 0.7)
        
        metrics = [
            (f"Student ID: {student_id}", self.colors["text"]),
            (f"Grade Level: {grade_level}", self.colors["text"]),
            (f"Learning Rate: {learning_rate:.0%}", self._get_color(learning_rate)),
            (f"Engagement: {engagement:.0%}", self._get_color(engagement)),
            (f"Fatigue: {fatigue:.0%}", self._get_color(1.0 - fatigue)),
            (f"Motivation: {motivation:.0%}", self._get_color(motivation)),
        ]
        
        y_offset = 155
        for label, color in metrics:
            text = self.font_small.render(label, True, color)
            self.screen.blit(text, (50, y_offset))
            y_offset += 30
        
        # Progress bar for current mastery
        mastery = state.get('current_mastery', 0.5)
        bar_bg = pygame.Rect(50, 320, 310, 15)
        bar_fill = pygame.Rect(50, 320, int(310 * mastery), 15)
        pygame.draw.rect(self.screen, self.colors["border"], bar_bg)
        pygame.draw.rect(self.screen, self.colors["success"], bar_fill)
        
        mastery_text = self.font_small.render(f"Current Mastery: {mastery:.0%}", True, self.colors["text"])
        self.screen.blit(mastery_text, (50, 340))
    
    def _draw_mastery_bars(self, state: Dict[str, Any]):
        """Draw mastery bars for all topics"""
        card = pygame.Rect(410, 100, 350, 250)
        pygame.draw.rect(self.screen, self.colors["bg_card"], card)
        pygame.draw.rect(self.screen, self.colors["border"], card, 2)
        
        title = self.font_medium.render("Topic Mastery", True, self.colors["primary"])
        self.screen.blit(title, (430, 115))
        
        topic_masteries = state.get('topic_masteries', [0] * 6)
        topic_names = state.get('topic_names', ["Cell Biology", "Genetics", "Evolution", "Ecology", "Human Body", "Plants"])
        
        y_offset = 155
        for i, (topic, mastery) in enumerate(zip(topic_names, topic_masteries)):
            topic_text = self.font_small.render(topic[:12], True, self.colors["text_muted"])
            self.screen.blit(topic_text, (430, y_offset))
            
            mastery_bar = pygame.Rect(550, y_offset + 2, 190, 12)
            mastery_fill = pygame.Rect(550, y_offset + 2, int(190 * mastery), 12)
            pygame.draw.rect(self.screen, self.colors["border"], mastery_bar)
            
            # Color based on mastery level
            if mastery >= 0.85:
                bar_color = self.colors["success"]
            elif mastery >= 0.6:
                bar_color = self.colors["primary"]
            elif mastery >= 0.3:
                bar_color = self.colors["warning"]
            else:
                bar_color = self.colors["danger"]
            
            pygame.draw.rect(self.screen, bar_color, mastery_fill)
            
            mastery_text = self.font_small.render(f"{mastery:.0%}", True, self.colors["text"])
            self.screen.blit(mastery_text, (745, y_offset))
            
            y_offset += 28
    
    def _draw_learning_progress(self, state: Dict[str, Any]):
        """Draw learning progress timeline"""
        card = pygame.Rect(30, 370, 730, 290)
        pygame.draw.rect(self.screen, self.colors["bg_card"], card)
        pygame.draw.rect(self.screen, self.colors["border"], card, 2)
        
        title = self.font_medium.render("Learning Progress", True, self.colors["primary"])
        self.screen.blit(title, (50, 385))
        
        # Topics progress bars
        topics_completed = state.get('topics_completed', 0)
        total_topics = state.get('total_topics', 6)
        
        progress_text = self.font_small.render(f"Topics Completed: {topics_completed}/{total_topics}", 
                                                True, self.colors["text"])
        self.screen.blit(progress_text, (50, 420))
        
        # Overall progress bar
        bar_bg = pygame.Rect(50, 445, 690, 20)
        bar_fill = pygame.Rect(50, 445, int(690 * topics_completed / max(total_topics, 1)), 20)
        pygame.draw.rect(self.screen, self.colors["border"], bar_bg)
        pygame.draw.rect(self.screen, self.colors["success"], bar_fill)
        
        # Percentage text
        percent_text = self.font_small.render(f"{int(topics_completed / max(total_topics, 1) * 100)}% Complete", 
                                               True, self.colors["text"])
        self.screen.blit(percent_text, (370, 448))
        
        # Engagement trend
        engagement = state.get('engagement', 0.7)
        step = state.get('step', 0)
        
        y_offset = 485
        engagement_title = self.font_small.render("Current Session Metrics:", True, self.colors["text"])
        self.screen.blit(engagement_title, (50, y_offset))
        
        y_offset += 25
        engagement_text = self.font_small.render(f"Engagement Level: {engagement:.0%}", 
                                                  True, self._get_color(engagement))
        self.screen.blit(engagement_text, (70, y_offset))
        
        y_offset += 25
        step_text = self.font_small.render(f"Session Steps: {step}", True, self.colors["text_muted"])
        self.screen.blit(step_text, (70, y_offset))
    
    def _draw_action_feedback(self, last_action: Optional[Dict], state: Dict[str, Any]):
        """Draw feedback for the last action"""
        card = pygame.Rect(780, 100, 390, 250)
        pygame.draw.rect(self.screen, self.colors["bg_card"], card)
        pygame.draw.rect(self.screen, self.colors["border"], card, 2)
        
        title = self.font_medium.render("AI Tutor Actions", True, self.colors["primary"])
        self.screen.blit(title, (800, 115))
        
        if last_action:
            action_name = last_action.get('action', 'None')
            if isinstance(action_name, int):
                action_names = ["Video Lesson", "Practice Quiz", "Problem Solving", "Review Summary",
                               "Adaptive Hint", "New Topic", "Remediation", "Peer Discussion", "Worked Example"]
                action_name = action_names[action_name] if action_name < len(action_names) else "Unknown"
            
            reward = last_action.get('reward', 0)
            
            action_text = self.font_small.render(f"Last Action: {action_name}", True, self.colors["text"])
            self.screen.blit(action_text, (800, 155))
            
            reward_color = self.colors["success"] if reward > 0 else self.colors["danger"]
            reward_text = self.font_small.render(f"Reward: {reward:+.1f}", True, reward_color)
            self.screen.blit(reward_text, (800, 185))
        else:
            waiting_text = self.font_small.render("Waiting for action...", True, self.colors["text_muted"])
            self.screen.blit(waiting_text, (800, 155))
        
        # Recommended actions section
        y_offset = 225
        rec_title = self.font_small.render("Recommended Next Actions:", True, self.colors["warning"])
        self.screen.blit(rec_title, (800, y_offset))
        
        mastery = state.get('current_mastery', 0.5)
        engagement = state.get('engagement', 0.7)
        
        recommendations = []
        if mastery < 0.4:
            recommendations = ["Video Lesson", "Adaptive Hint"]
        elif mastery < 0.7:
            recommendations = ["Practice Quiz", "Peer Discussion"]
        elif mastery >= 0.7:
            recommendations = ["New Topic", "Problem Solving"]
        
        if engagement < 0.4:
            recommendations = ["Peer Discussion", "Worked Example"]
        
        y_offset += 25
        for rec in recommendations[:2]:
            rec_text = self.font_small.render(f"- {rec}", True, self.colors["text_muted"])
            self.screen.blit(rec_text, (820, y_offset))
            y_offset += 22
    
    def _draw_recommendations(self, state: Dict[str, Any]):
        """Draw AI recommendations"""
        card = pygame.Rect(780, 370, 390, 290)
        pygame.draw.rect(self.screen, self.colors["bg_card"], card)
        pygame.draw.rect(self.screen, self.colors["border"], card, 2)
        
        title = self.font_medium.render("AI Insights", True, self.colors["primary"])
        self.screen.blit(title, (800, 385))
        
        mastery = state.get('current_mastery', 0)
        engagement = state.get('engagement', 0)
        fatigue = state.get('fatigue_level', 0)
        topics_completed = state.get('topics_completed', 0)
        total_topics = state.get('total_topics', 6)
        
        insights = []
        
        if mastery >= 0.85:
            insights.append("Excellent progress on current topic")
            insights.append("Ready to advance to next topic")
        elif mastery >= 0.6:
            insights.append("Good understanding developing")
            insights.append("Continue with practice activities")
        elif mastery >= 0.3:
            insights.append("Needs more practice")
            insights.append("Try different learning activities")
        else:
            insights.append("Struggling with this topic")
            insights.append("Review fundamentals with hints")
        
        if engagement < 0.4:
            insights.append("Engagement is low - try gamified activities")
        
        if fatigue > 0.7:
            insights.append("Student shows signs of fatigue")
            insights.append("Recommend taking a short break")
        
        if topics_completed == total_topics:
            insights = ["CONGRATULATIONS!", "All topics mastered!", "Excellent learning progress!"]
        
        y_offset = 420
        for insight in insights[:5]:
            insight_text = self.font_small.render(insight, True, self.colors["warning"])
            self.screen.blit(insight_text, (800, y_offset))
            y_offset += 25
    
    def _get_color(self, value: float) -> tuple:
        """Get color based on value (red to green gradient)"""
        if value >= 0.7:
            return self.colors["success"]
        elif value >= 0.4:
            return self.colors["warning"]
        else:
            return self.colors["danger"]
    
    def close(self):
        pygame.quit()