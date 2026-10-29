"""
This module contains the DePlotCaptioner class for two-stage chart captioning.
"""

# Standard libraries
import warnings
from pathlib import Path
from typing import Union

# 3pps
import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

# Own modules
from models.local_llm import generate as llm_generate


# System prompt for the captioning stage (DePlot → Qwen)
_CAPTIONING_SYSTEM_PROMPT: str = "You are a professional and precise data analyst assistant."

_USER_PROMPT: str = """\
You are an expert data analyst. Read the following data table extracted from a chart \
and write a fluent narrative paragraph summarizing it.
Follow this strict structure:
1. Start by saying "The chart shows..." and explain what the data is about.
2. Highlight the overall trend.
3. Explicitly state the highest peak and the lowest value.
Do not invent any data or add external information.

Data Table:
{table}

Narrative summary:"""


def _pick_torch_device() -> str:
    """Pick CUDA only when the current torch build can run kernels."""

    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return "cuda"
        except Exception as exc:
            print(f"[DePlot] CUDA detected but unusable ({exc}); using CPU.")
    return "cpu"


class DePlotCaptioner:
    """
    Two-stage chart captioner: DePlot (image→table) followed by Qwen (table→caption).

    Attr:
        deplot_model_id:    HuggingFace model ID for DePlot.
        qwen_model_id:      HuggingFace model ID for the Qwen LLM.
        max_table_tokens:   Maximum new tokens for DePlot generation.
        max_caption_tokens: Maximum new tokens for Qwen generation.
        temperature:        Sampling temperature for the Qwen caption.
        device:             Torch device string ('cuda' or 'cpu').
    """

    def __init__(
        self,
        deplot_model_id:    str   = "google/deplot",
        qwen_model_id:      str   = "Qwen/Qwen2.5-1.5B-Instruct",
        max_table_tokens:   int   = 512,
        max_caption_tokens: int   = 200,
        temperature:        float = 0.3,
    ) -> None:
        """
        This function initialises the DePlotCaptioner with model IDs and generation parameters.

        Args:
            deplot_model_id:    HuggingFace model ID for DePlot.
            qwen_model_id:      HuggingFace model ID for the Qwen LLM.
            max_table_tokens:   Maximum new tokens for DePlot generation.
            max_caption_tokens: Maximum new tokens for Qwen generation.
            temperature:        Sampling temperature for the Qwen caption.
        """

        self.deplot_model_id:    str   = deplot_model_id
        self.qwen_model_id:      str   = qwen_model_id
        self.max_table_tokens:   int   = max_table_tokens
        self.max_caption_tokens: int   = max_caption_tokens
        self.temperature:        float = temperature
        self.device:             str   = _pick_torch_device()

        self._processor: Pix2StructProcessor               | None = None
        self._deplot:    Pix2StructForConditionalGeneration | None = None

        return None

    def load_models(self) -> None:
        """
        This function loads DePlot weights and warms up the Qwen singleton.

        Must be called from the main thread before any captioning.
        """

        if self._deplot is None:
            print(f"[DePlot] Loading DePlot ({self.deplot_model_id})…")
            self._processor = Pix2StructProcessor.from_pretrained(self.deplot_model_id)
            self._deplot    = Pix2StructForConditionalGeneration.from_pretrained(
                self.deplot_model_id
            ).to(self.device)
            self._deplot.eval()
            # Prevent conflict between model's default max_length and max_new_tokens
            self._deplot.generation_config.max_length = None
            print("[DePlot] DePlot ready.")

        # Warm up Qwen singleton before any threading starts
        from models.local_llm import get_pipeline
        get_pipeline(model_id=self.qwen_model_id)
        print("[DePlot] Qwen ready.")

        return None

    def caption(self, image_path: Union[str, Path]) -> tuple[str, str]:
        """
        This function runs the full two-stage pipeline on a chart image.

        Args:
            image_path: Path to the chart image file.

        Returns:
            A tuple (table, narrative) where table is the raw DePlot output and
            narrative is the Qwen-generated caption.
        """

        from PIL import Image

        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        # Stage 1: image → structured table via DePlot
        table: str = self.generate_table(image)

        # Stage 2: table → narrative caption via Qwen
        narrative: str = self.generate_caption(table)

        return table, narrative

    def generate_table(self, image) -> str:
        """
        This function runs DePlot to extract a structured data table from an image.

        Args:
            image: A PIL Image in RGB mode.

        Returns:
            A string containing the extracted data table.
        """

        print("[DePlot] Extracting data table…")

        inputs = self._processor(
            images=image,
            text="Generate underlying data table of the figure below:",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*max_new_tokens.*max_length.*")
            preds = self._deplot.generate(**inputs, max_new_tokens=self.max_table_tokens)

        raw:   str = self._processor.decode(preds[0], skip_special_tokens=True)
        table: str = raw.replace("<0x0A>", "\n").strip()

        first_line: str = table.splitlines()[0] if table else "(empty)"
        print(f"[DePlot] Table ready ({len(table)} chars). First line: {first_line}")

        return table

    def generate_caption(self, table: str) -> str:
        """
        This function generates a narrative caption from a structured data table.

        Args:
            table: The data table string produced by DePlot.

        Returns:
            A fluent narrative string summarising the chart data.
        """

        print("[DePlot] Generating narrative caption…")

        narrative: str = llm_generate(
            messages=[
                {"role": "system", "content": _CAPTIONING_SYSTEM_PROMPT},
                {"role": "user",   "content": _USER_PROMPT.format(table=table)},
            ],
            model_id=self.qwen_model_id,
            max_new_tokens=self.max_caption_tokens,
            temperature=self.temperature,
        )

        print(f"[DePlot] Caption ready ({len(narrative)} chars).")

        return narrative
