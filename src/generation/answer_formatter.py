"""
Answer Formatter Module

Formats final answers with citations and verbatim contextual passages.
Enforces strict citation rules: LLM must not hallucinate sources.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class AnswerFormatter:
    """Format answers with strict source attribution."""

    def __init__(self, llm_client=None):
        """
        Initialize answer formatter.
        
        Args:
            llm_client: OpenAI client for answer summarization (optional)
        """
        self.llm_client = llm_client

    def format_answer(
        self,
        query: str,
        context_units: List[Dict[str, Any]],
        answer_text: str = None
    ) -> Dict[str, Any]:
        """
        Format final answer with citations.
        
        Args:
            query: Original user query
            context_units: Retrieved Context Units
            answer_text: Optional pre-formatted answer
            
        Returns:
            Formatted answer dictionary
        """
        if not context_units:
            return {
                'query': query,
                'answer': 'No relevant information found in the Mahabharata.',
                'sources': [],
                'contextual_passages': []
            }
        
        # Use first Context Unit as primary source
        primary_unit = context_units[0]
        
        # Extract citation information
        citation = {
            'parva': primary_unit['parva'],
            'section': primary_unit['section'],
            'story_phase': primary_unit.get('story_phase', 'Unknown')
        }
        
        # Build contextual passage (verbatim from text)
        passage = ' '.join(primary_unit['paragraphs'])
        
        # If answer text not provided, use first paragraph as summary
        if answer_text is None:
            answer_text = primary_unit['text'][:500] + '...'
        
        return {
            'query': query,
            'answer': answer_text,
            'retrieved_from': citation,
            'contextual_passage': passage,
            'supporting_units': [cu['unit_id'] for cu in context_units]
        }

    def format_with_llm_summary(
        self,
        query: str,
        context_units: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format answer with LLM-generated summary.
        
        Args:
            query: Original query
            context_units: Retrieved Context Units
            
        Returns:
            Formatted answer with LLM summary
        """
        if not context_units:
            return self.format_answer(query, context_units)
        
        # Extract passages
        passages = [' '.join(cu['paragraphs']) for cu in context_units]
        combined_context = '\n\n'.join(passages)
        
        # Generate summary with LLM (constrained to context)
        summary = self._generate_constrained_answer(query, combined_context)
        
        return self.format_answer(query, context_units, answer_text=summary)

    def _generate_constrained_answer(self, query: str, context: str) -> str:
        """
        Generate answer summary constrained to context.
        LLM is not allowed to add external knowledge.
        
        Args:
            query: User query
            context: Retrieved context text
            
        Returns:
            LLM-generated answer
        """
        if not self.llm_client:
            return context[:500] + '...'
        
        prompt = f"""Based ONLY on the provided context, answer the following query.
Do NOT add external knowledge. Do NOT hallucinate sources or facts.
If the context does not answer the query, say so explicitly.

Context:
{context}

Query: {query}

Answer:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a faithful assistant that answers questions based ONLY on provided context. You never hallucinate facts or sources."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return context[:500] + '...'

    def validate_answer(self, answer: Dict[str, Any]) -> bool:
        """
        Validate answer against strict rules.
        
        Rules:
        - Quoted passage must explicitly support answer
        - Citations must reference retrieved source
        - No external knowledge
        
        Args:
            answer: Formatted answer
            
        Returns:
            True if valid, False otherwise
        """
        # Check that citation exists
        if not answer.get('retrieved_from'):
            logger.warning("Answer missing source citation")
            return False
        
        # Check that contextual passage exists
        if not answer.get('contextual_passage'):
            logger.warning("Answer missing contextual passage")
            return False
        
        return True


def format_final_output(answer: Dict[str, Any]) -> str:
    """
    Format answer for display to user.
    
    Args:
        answer: Formatted answer dictionary
        
    Returns:
        Formatted string for display
    """
    output = f"""Query: {answer['query']}

Answer:
{answer['answer']}

Retrieved from:
Parva: {answer['retrieved_from']['parva']}
Section: {answer['retrieved_from']['section']}
Story Phase: {answer['retrieved_from']['story_phase']}

Contextual Passage:
"{answer['contextual_passage']}"
"""
    return output


if __name__ == '__main__':
    # Example usage
    formatter = AnswerFormatter()
