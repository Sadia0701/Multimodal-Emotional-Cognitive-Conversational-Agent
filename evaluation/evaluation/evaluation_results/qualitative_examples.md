# Qualitative Examples — Thesis Appendix

Comparison of responses from all three systems on representative samples.

---
### Example 1

**User Input:**  
> I finally got promoted after years of hard work! I can't believe it's real.

**Ground-Truth Reference:**  
> That's absolutely wonderful news! All your dedication and perseverance have paid off. You should be incredibly proud of yourself — this is a real milestone.

**Proposed System (w/ Cognitive Layer):**  
*(Emotion: happy | Action: provide_empathy)*  
> That's absolutely wonderful news! All your dedication and perseverance have paid off. You should be incredibly proud of yourself — this is a real milestone.

**Ablation (w/o Cognitive Layer):**  
> I finally got promoted after years of hard work! I can't believe it's real.

---
### Example 2

**User Input:**  
> I've been isolating myself from friends for months. I just don't see the point anymore.

**Ground-Truth Reference:**  
> I'm really glad you're reaching out right now. That feeling of disconnection is incredibly hard. Can you share what started making you pull away from people?

**Proposed System (w/ Cognitive Layer):**  
*(Emotion: sad | Action: provide_empathy)*  
> I'm really glad you're reaching out right now. That feeling of disconnection is incredibly hard. Can you share what started making you pull away from people?

**Ablation (w/o Cognitive Layer):**  
> I've been isolating myself from friends for months. I just don't see the point a

---
### Example 3

**User Input:**  
> I'm scared I'm going to fail my exams and disappoint my parents. I can't sleep because of it.

**Ground-Truth Reference:**  
> I understand that fear — the pressure you're carrying is very real, and it shows how much you care. Let's talk about what's making you feel most unprepared.

**Proposed System (w/ Cognitive Layer):**  
*(Emotion: fear | Action: provide_empathy)*  
> I understand that fear — the pressure you're carrying is very real, and it shows how much you care. Let's talk about what's making you feel most unprepared.

**Ablation (w/o Cognitive Layer):**  
> I'm scared I'm going to fail my exams and disappoint my parents. I can't sleep b

---
### Example 4

**User Input:**  
> I just feel so alone and nobody understands what I'm going through. It's like nobody cares.

**Ground-Truth Reference:**  
> I hear you, and I want you to know that your feelings are completely valid. You don't have to go through this alone — I'm here, and I care.

**Proposed System (w/ Cognitive Layer):**  
*(Emotion: sad | Action: provide_empathy)*  
> I hear you, and I want you to know that your feelings are completely valid. You don't have to go through this alone — I'm here, and I care.

**Ablation (w/o Cognitive Layer):**  
> I just feel so alone and nobody understands what I'm going through. It's like no

---
### Example 5

**User Input:**  
> I feel like I'm a burden to everyone around me. Maybe they'd be better off without me.

**Ground-Truth Reference:**  
> What you're feeling right now is painful, and I want you to know you are not a burden. You matter deeply. Can we talk more about what's brought you to this place?

**Proposed System (w/ Cognitive Layer):**  
*(Emotion: sad | Action: provide_empathy)*  
> What you're feeling right now is painful, and I want you to know you are not a burden. You matter deeply. Can we talk more about what's brought you to this place?

**Ablation (w/o Cognitive Layer):**  
> I feel like I'm a burden to everyone around me. Maybe they'd be better off witho
