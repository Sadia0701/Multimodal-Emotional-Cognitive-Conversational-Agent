# Qualitative Examples — Thesis Appendix

Comparison of responses from all three systems on representative samples.

---
### Example 1

**User Input:**  
> Nobody at school knows I'm struggling. I act like everything is fine all the time.

**Ground-Truth Reference:**  
> Carrying that mask every day is exhausting work. I'm glad you can take it off here. What would it feel like if someone at school actually knew?

**Proposed System (w/ Cognitive Layer):**  
*(Emotion: sad | Action: provide_empathy)*  
> Carrying that mask every day is exhausting work. I'm glad you can take it off here. What would it feel like if someone at school actually knew?

**Ablation (w/o Cognitive Layer):**  
> Nobody at school knows I'm struggling. I act like everything is fine all the tim

---
### Example 2

**User Input:**  
> I've been having panic attacks again and I don't know what triggers them anymore.

**Ground-Truth Reference:**  
> Panic attacks can be frightening and exhausting. I'm really glad you're talking about it. Can you tell me more about when they usually happen?

**Proposed System (w/ Cognitive Layer):**  
*(Emotion: fear | Action: provide_empathy)*  
> Panic attacks can be frightening and exhausting. I'm really glad you're talking about it. Can you tell me more about when they usually happen?

**Ablation (w/o Cognitive Layer):**  
> I've been having panic attacks again and I don't know what triggers them anymore

---
### Example 3

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
### Example 4

**User Input:**  
> I've been drinking more than I should to cope with the stress. I know it's not healthy.

**Ground-Truth Reference:**  
> I really appreciate you trusting me with that. Recognising it yourself is already an important step. What kind of stress has been pushing you to reach for it?

**Proposed System (w/ Cognitive Layer):**  
*(Emotion: fear | Action: provide_empathy)*  
> I really appreciate you trusting me with that. Recognising it yourself is already an important step. What kind of stress has been pushing you to reach for it?

**Ablation (w/o Cognitive Layer):**  
> I've been drinking more than I should to cope with the stress. I know it's not h

---
### Example 5

**User Input:**  
> I'm terrified about my health results coming back. I keep imagining the worst.

**Ground-Truth Reference:**  
> Waiting for health news is one of the hardest things to sit with. Your feelings make complete sense. I'm here with you through this.

**Proposed System (w/ Cognitive Layer):**  
*(Emotion: fear | Action: provide_empathy)*  
> Waiting for health news is one of the hardest things to sit with. Your feelings make complete sense. I'm here with you through this.

**Ablation (w/o Cognitive Layer):**  
> I'm terrified about my health results coming back. I keep imagining the worst.
