from datetime import datetime
from typing import Any, Dict, List, Optional
import json

from chainlit.context import context_var
from chainlit.element import Text
from chainlit.step import Step, StepType

from llama_index.core.callbacks import TokenCountingHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

DEFAULT_IGNORE = [
    CBEventType.CHUNKING,
    CBEventType.SYNTHESIZE,
    CBEventType.EMBEDDING,
    CBEventType.NODE_PARSING,
    CBEventType.QUERY,
    CBEventType.TREE,
    CBEventType.RETRIEVE,   
    CBEventType.TEMPLATING,
    CBEventType.LLM,
    CBEventType.SUB_QUESTION,
    # CBEventType.RERANKING
]


class CustomLlamaIndexCallbackHandler(TokenCountingHandler):
    """Base callback handler that can be used to track event starts and ends."""

    steps: Dict[str, Step]

    def __init__(
        self,
        event_starts_to_ignore: List[CBEventType] = DEFAULT_IGNORE,
        event_ends_to_ignore: List[CBEventType] = DEFAULT_IGNORE,
    ) -> None:
        """Initialize the base callback handler."""
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
        )
        self.context = context_var.get()

        self.steps = {}

    def _get_parent_id(self, event_parent_id: Optional[str] = None) -> Optional[str]:
        if event_parent_id and event_parent_id in self.steps:
            return event_parent_id
        elif self.context.current_step:
            return self.context.current_step.id
        elif self.context.session.root_message:
            return self.context.session.root_message.id
        else:
            return None

    def _restore_context(self) -> None:
        """Restore Chainlit context in the current thread

        Chainlit context is local to the main thread, and LlamaIndex
        runs the callbacks in its own threads, so they don't have a
        Chainlit context by default.

        This method restores the context in which the callback handler
        has been created (it's always created in the main thread), so
        that we can actually send messages.
        """
        context_var.set(self.context)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        self._restore_context()
        step_type: StepType = "undefined"
        if event_type == CBEventType.RETRIEVE:
            step_type = "retrieval"

        elif event_type == CBEventType.RERANKING:
            step_type = "reranking"

        elif event_type == CBEventType.LLM:
            step_type = "llm"
        
        elif event_type == CBEventType.SYNTHESIZE:
            step_type = "synthetizer"

        elif event_type == CBEventType.SUB_QUESTION:
            step_type = "sub_question"

        elif event_type == CBEventType.QUERY:
            step_type = "query"
        
        else:
            print(f"[LLamaIndexCallbackHandler] - Missed {event_type}")
            return event_id

        step = Step(
            name=event_type.value,
            type=step_type,
            parent_id=self._get_parent_id(parent_id),
            id=event_id,
            disable_feedback=False,
            show_input=True
        )
        print(f"START STEP --> {step.to_dict()}")
        self.steps[event_id] = step
        step.start = datetime.utcnow().isoformat()
        step.input = payload.get(EventPayload.QUERY_STR) if (event_type == CBEventType.RERANKING or event_type == CBEventType.RETRIEVE) else (payload or {})
        self.context.loop.create_task(step.send())
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        step = self.steps.get(event_id, None)

        if payload is None or step is None:
            return

        self._restore_context()
        step.end = datetime.utcnow().isoformat()

        if event_type == CBEventType.RETRIEVE:
            sources = payload.get(EventPayload.NODES)

            if sources:
                if any([source.score > 1 for source in sources]):
                    step.name = "BM25 Retrieval"
                else:
                    step.name = "Vector Retrieval"
                source_refs = "\, ".join([f"Fonte {idx}" for idx, _ in enumerate(sources)])

                step.elements = [Text(name=f"Fonte {idx}", content=self.format_source(source.node) or "Empty node",) for idx, source in enumerate(sources)]
                step.input = "**Input query**: " + (step.input or "Empty Input")
                step.output = f"**Recuperate le seguenti fonti**: {source_refs}"
        
        elif event_type == CBEventType.QUERY:
            response = payload.get(EventPayload.RESPONSE)

            if response.metadata and ("selector_result" in response.metadata):
                selector_result = response.metadata['selector_result'].selections[0]
                step.input = "**Input query**: " + json.loads(step.input)['query_str']
                step.output = f"**LLM Output**: Selecting query engine {selector_result.index + 1}: {selector_result.reason}"
                step.name = "Tool Selection"
            elif response.source_nodes:
                source_refs = "\, ".join([f"Fonte {idx}" for idx, _ in enumerate(response.source_nodes)])
                step.elements = [Text(name=f"Fonte {idx}", content=self.format_source(source.node) or "Empty node",) for idx, source in enumerate(response.source_nodes)]

                step.input = "**Input query for selected Tool**: " + json.loads(step.input)['query_str']
                step.output = f"**Tool Sources**: {source_refs}"
                step.name = "Retrieval"
        
        elif event_type == CBEventType.SYNTHESIZE:
            response = payload.get(EventPayload.RESPONSE)
            
            if response:
                step.input = "Input query: " + json.loads(step.input)['query_str']
                # step.elements = [Text(name=f"Intermediate Answer", content=response.response_txt or "Empty Answer")]
                step.output = f"Generata una risposta intermedia"
        
        elif event_type == CBEventType.SUB_QUESTION:
            subquestion = payload.get(EventPayload.SUB_QUESTION)
            
            if subquestion:
                step.input = "Input query: " + json.loads(step.input)['query_str']
                step.output = f"Generata la seguente SubQuestion" + "\nIntermediate answer"
                step.elements = [
                    Text(name=f"SubQuestion", content=subquestion.sub_q or "Empty SubQuestion"), 
                    Text(name=f"Intermediate Answer", content=subquestion.answer or "Empty Answer")
                    ]

        elif event_type == CBEventType.LLM:
            formatted_prompt = payload.get(EventPayload.PROMPT)
            completion = payload.get(EventPayload.COMPLETION)

            if completion:
                if formatted_prompt.startswith("Given a conversation"):
                    step.name = "LLM Conversation Condensation"
                elif formatted_prompt.startswith("Some choices are given below"):
                    step.name = "LLM Tool Selection"
                elif formatted_prompt.startswith("Context information is below"):
                    step.name = "LLM Intermediate Answer"
                elif formatted_prompt.startswith("Given a user question"):
                    step.name = "Sub-question Generation"
                else:
                    step.name = "LLM Generation"
                
                step.input = "Input Prompt"
                step.output = "LLM Output"
                step.elements = [
                    Text(name=f"Input Prompt", content= f"```\n{formatted_prompt}\n```" if formatted_prompt else "Empty Prompt"), 
                    Text(name=f"LLM Output", content=f"```\n{completion.text}\n```" if completion.text else "No Content")
                    ]
                        
        elif event_type == CBEventType.RERANKING:
            sources = payload.get(EventPayload.NODES)

            if sources:
                source_refs = "\, ".join([f"Fonte {idx}" for idx, source in enumerate(sources)])

                step.elements = [Text(name=f"Fonte {idx}", content=self.format_source(source.node) or "Empty node", display="side") for idx, source in enumerate(sources)]
                step.input = "**Input query**: " + (step.input or "Empty Input")
                step.output = f"**Recuperate le seguenti fonti**: {source_refs}"
                step.name = "Relevant Source Reranking"

        else:
            step.output = payload

        print(f"END STEP --> {step.to_dict()}")
        if (step.name == "LLM Intermediate Answer") or (step.name == "LLM Generation"):
            self.context.loop.create_task(step.remove())
        else:
            self.context.loop.create_task(step.update())
        self.steps.pop(event_id, None)

    def _noop(self, *args, **kwargs):
        pass

    def format_source(self, source_node):
        output_str = ""
        for k,v in source_node.metadata.items():
            output_str += f"**{k}**: {v}\n"
        output_str += f'\n\n**Text**\n{source_node.text}'
        return output_str


    start_trace = _noop
    end_trace = _noop