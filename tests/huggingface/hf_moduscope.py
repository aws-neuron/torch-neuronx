"""HuggingFace model wrapper for ModuScope.

This module provides classes and utilities for working with HuggingFace models in ModuScope.

Task Mappings:
--------------
Maps a task type to Hugging Face API tags to find models corresponding to that task type.

The TASK_MAPPINGS dictionary provides the necessary mappings between task types (e.g., 'CausalLM')
and their corresponding Hugging Face components. Each task entry contains:

- auto_class: The AutoModel class used to load pretrained models for this task
- mapping: The dictionary from modeling_auto that gives us the architectures for a task type
- pipeline_tag: The Hugging Face pipeline tag string used for searching models on the Hub API

Notes:

Why are both mapping and pipeline_tag needed?

- mapping: Used to retrieve architecture names (e.g., 'gpt2') for a task type
- pipeline_tag: Used when searching the Hugging Face Hub API for models for a particular task

Why does this have to be hardcoded?

We maintain both entries as there's no reliable way to automatically derive
one from the other due to inconsistent naming patterns (e.g. 'CausalLM' has
'text-generation' as its task type).

What tasks does this mapping currently support?

We currently only include the tasks that are associated with language models
and cover the main language model architectures. This approach could be extended
to include multimodal AutoModel classes and pipeline tags in the future (e.g.
an auto class of 'AutoModelForImageClassification' and a pipeline tag of
'image-classification'.

For more informaton about Auto Classes, see:
https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes

For more information about pipeline tags, see:
https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task
"""

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from filelock import FileLock
from huggingface_hub import HfApi, snapshot_download
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForMaskGeneration,
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTextEncoding,
    AutoModelForTokenClassification,
)
from transformers.models.auto import modeling_auto

# from neuron_gspmd_tests.modeling.modules.module_base import NotAnnotated
# from neuron_gspmd_tests.registry.component_registry import get_registry, register_model

logger = logging.getLogger(__name__)

TASK_MAPPINGS = {
    # CausalLM, MaskedLM, Seq2SeqLM cover most common language model architectures
    "CausalLM": {
        "auto_class": AutoModelForCausalLM,
        "mapping": modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        "pipeline_tag": "text-generation",
    },
    "MaskedLM": {
        "auto_class": AutoModelForMaskedLM,
        "mapping": modeling_auto.MODEL_FOR_MASKED_LM_MAPPING_NAMES,
        "pipeline_tag": "fill-mask",
    },
    "Seq2SeqLM": {
        "auto_class": AutoModelForSeq2SeqLM,
        "mapping": modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        "pipeline_tag": "text2text-generation",
    },
    # Other task-specific heads can also be added to base models
    "MultipleChoice": {
        "auto_class": AutoModelForMultipleChoice,
        "mapping": modeling_auto.MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
        "pipeline_tag": "multiple-choice",
    },
    "TokenClassification": {
        "auto_class": AutoModelForTokenClassification,
        "mapping": modeling_auto.MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
        "pipeline_tag": "token-classification",
    },
    "QuestionAnswering": {
        "auto_class": AutoModelForQuestionAnswering,
        "mapping": modeling_auto.MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
        "pipeline_tag": "question-answering",
    },
    # Default fallback
    "default": {"auto_class": AutoModel, "mapping": {}, "pipeline_tag": None},
}


class HuggingFaceModelManager:
    """
    Manager class for HuggingFace models in ModuScope.

    This class provides methods to download, initialize, and create wrapper classes
    for HuggingFace models to be used with ModuScope.

    Models can be initialized by downloading the model config file or by downloading the
    model weights from HuggingFace. Set download_mode='config_only' or 'full_model' in the
    Hugging Face task config file. Defaults to 'config_only'.
    """

    # To avoid duplicate models being registered by parallel processes
    from typing import ClassVar

    _created_model_classes: ClassVar[dict] = {}

    # Class dict for model classes and configs per task type
    _cached_model_classes: ClassVar[dict] = {}
    _cached_model_configs: ClassVar[dict] = {}

    # Shared artifact directory
    _artifact_dir = None

    @staticmethod
    def create_model_config_candidates(
        task_type: str = "CausalLM",
        download_mode: str = "config_only",
        artifact_directory: str | None = None,
    ) -> tuple[list[tuple[str]], list[tuple[str | None]]]:
        """
        Create model candidate tuples and model config tuples for the model grid config.

        Args:
            task_type: The type of task (e.g., "CausalLM")
            download_mode: Whether to download "config_only" or "full_model"
            artifact_directory: Directory to save artifacts. Obtained from the task config
                and used to store model configs.

        Returns:
            Tuple[List[Tuple[str]], List[Tuple[Optional[str]]]]:
                A tuple containing:
                    - List of model candidates tuples
                    - List of model configs tuples (some may be None)
        """

        # Check if we've already cached configs for this task type
        cache_key = f"{task_type}_{download_mode}_{artifact_directory}"
        if cache_key in HuggingFaceModelManager._cached_model_configs:
            logger.info(f"Using cached model configs for {task_type}")
            return HuggingFaceModelManager._cached_model_configs[cache_key]

        # Get model classes for this task type
        classes = HuggingFaceModelManager.get_model_classes(
            task_type=task_type, download_mode=download_mode, artifact_directory=artifact_directory
        )

        candidates = []
        configs = []

        # Iterate through the classes and create the tuples for the model grid config
        for class_name, _, model_id in classes:
            # Add the model candidate (loss is handled in run_model_step)
            candidates.append((class_name,))

            # Add the model config
            model_config = HuggingFaceModelManager.Loader.get_or_download_model_config(
                model_id, artifact_directory
            )
            if model_config is None:
                # Set config as None if not found to ensure lengths match
                # This ensures proper pairing in the grid config
                logger.warning(f"Failed to load config for {model_id}, falling back to None")
            configs.append((model_config,))

        result = (candidates, configs)
        HuggingFaceModelManager._cached_model_configs[cache_key] = result
        logger.info(
            f"Created the following model candidates for task type {task_type}: {candidates}"
        )
        logger.info(f"Created the following model configs for task type {task_type}: {configs}")
        logger.info(f"Created {len(candidates)} model candidates for task type {task_type}")
        return result

    @staticmethod
    def get_model_classes(
        task_type: str = "CausalLM",
        download_mode: str = "config_only",
        artifact_directory: str | None = None,
    ) -> list[tuple[str, type, str]]:
        """
        Obtains the models and creates their respective classes for this task type.

        Uses helper methods to find and create appropriate model classes.

        Args:
            task_type: Task type for the models (e.g., 'CausalLM')
            download_mode: Whether to download 'config_only' or 'full_model'
            artifact_directory: Directory to save artifacts

        Returns:
            list[tuple[str, type, str]]: List of tuples with class name, class object,
            and full model ID
        """
        # Check if we've already cached models for this task type
        cache_key = f"{task_type}_{download_mode}"
        if cache_key in HuggingFaceModelManager._cached_model_classes:
            logger.info(f"Using cached model classes for {task_type}")
            return HuggingFaceModelManager._cached_model_classes[cache_key]

        created_classes = []

        # Get the task mapping for the specified task type
        task_info = TASK_MAPPINGS.get(task_type)
        if not task_info:
            supported_tasks = list(TASK_MAPPINGS.keys())
            logger.warning(
                f"Unknown task type: {task_type}. "
                f"Currently supported task types are: {supported_tasks}. "
                "To add a new task type, update TASK_MAPPINGS dictionary with appropriate "
                "auto_class, mapping, and pipeline_tag values in task_mappings.py"
            )
            return []

        # Get architectures for this task mapping
        architectures = list(task_info["mapping"].keys())
        all_models = []

        # Get the popular models for this architecture
        for arch in architectures:
            models = HuggingFaceModelManager.get_popular_models_for_architecture(
                arch, task_type, limit=1
            )
            all_models.extend(models)
        logger.info(f"Found the following models from the HF API: {all_models}")
        logger.info(f"Found {len(all_models)} models from the HF API")

        # Create the wrapper classes for each of these models
        for model_info in all_models:
            full_id = model_info["full_id"]
            camel_name = model_info["camel_name"]

            # Create the model class with the appropriate parameters
            class_obj = HuggingFaceModelManager.create_model_class_with_full_id(
                camel_name=camel_name,
                full_model_id=full_id,
                task_type=task_type,
                default_download_mode=download_mode,
                default_artifact_dir=artifact_directory,
            )

            # Add the class to our list
            class_name = f"Auto{camel_name}For{task_type}"
            created_classes.append((class_name, class_obj, full_id))

        logger.info(f"Created the following classes for task type {task_type}: {created_classes}")
        logger.info(f"Created {len(created_classes)} total classes for task type {task_type}")

        # Cache our created classes
        HuggingFaceModelManager._cached_model_classes[cache_key] = created_classes
        return created_classes

    @staticmethod
    def get_popular_models_for_architecture(
        architecture_name: str, task_type: str, limit: int = 1
    ) -> list[dict[str, str]]:
        """
        Find popular models for a given architecture from the Hugging Face Hub.

        Calls the Hugging Face API to find models under the given architecture name with the
        given task type. Obtains the most popular full model ids with these parameters.
        Converts the full model id used in Hugging Face to match common class name convention.

        For example:
            - Input: architecture_name='gpt2', task_type='CausalLM'
            - Outputs:
                - full_id: 'openai-community/gpt2'
                - camel_name: 'OpenaiCommunityGpt2'

        Args:
            architecture_name: Base architecture name (e.g., 'gpt2')
            task_type: Task type (e.g., 'CausalLM')
            limit: Maximum number of models to return per architecture

        Returns:
            List of dictionaries with full_id and camel_name for models
        """
        api = HfApi()

        # Search for models matching the pipeline tag (corresponds to task type)
        # and architecture name
        pipeline_tag = TASK_MAPPINGS.get(task_type, TASK_MAPPINGS["default"])["pipeline_tag"]

        try:
            # import pdb; pdb.set_trace()
            # TODO: Look into whether pipeline tag and limit are necessary in new API
            # Request double the limit since some models might be filtered out by our matching logic
            if "whisper" in architecture_name:
                return []
            models = api.list_models(
                model_name=architecture_name,
                task=pipeline_tag if pipeline_tag else None,
                library="transformers",
                sort="downloads",
                direction=-1,
                limit=limit * 2,
            )

            results = []
            for model in models:
                model_id = model.id
                model_id_lower = model_id.lower()

                # Confirm architecture name appears in model ID
                # This ensures we get models specifically for this architecture
                # Example: for 'gpt2', we want 'openai-community/gpt2' but not 'microsoft/DialoGPT'
                if architecture_name.lower() not in model_id_lower:
                    logger.debug(
                        f"Skipping model {model_id} as it doesn't "
                        f" match architecture {architecture_name}"
                    )
                    continue

                # Split organization name from model name if present
                if "/" in model_id:
                    org, model_name = model_id.split("/", 1)
                else:
                    org = ""
                    model_name = model_id

                # Process organization name as camel name
                if org:
                    clean_org = re.sub(r"[^\w\s-]", "", org)
                    org_parts = re.findall(r"[A-Za-z0-9]+", clean_org)
                    org_camel = "".join(part.capitalize() for part in org_parts)
                else:
                    org_camel = ""

                # Process model name as camel name
                clean_name = re.sub(r"[^\w\s-]", "", model_name)
                parts = re.findall(r"[A-Za-z0-9]+", clean_name)
                model_camel = "".join(part.capitalize() for part in parts)

                # Combine organization and model names for the final camel case name
                camel_name = f"{org_camel}{model_camel}"

                # Add our full model id and camel name to results
                results.append({"full_id": model_id, "camel_name": camel_name})
                logger.info(f"Found model {model_id} and created camel name {camel_name}")

                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            logger.warning(f"Error fetching models for {architecture_name}: {e}")
            return []

    @staticmethod
    def create_model_class_with_full_id(
        camel_name: str,
        full_model_id: str,
        task_type: str,
        default_download_mode: str = "config_only",
        default_artifact_dir: str | None = None,
    ) -> type:
        """
        Wraps and creates a Hugging Face model class using a full model id.

        This function dynamically creates a class that wraps a HuggingFace model
        for use with ModuScope. The created class will handle initialization,
        sharding strategies, and forward pass for the model. Currently inherits
        from NotAnnotated class since no sharding strategies are defined.

        'Auto{ModelName}For{TaskType}' is used for the class name (e.g., 'AutoGpt2ForCausalLM').

        Args:
            camel_name: CamelCase model name to use in the generated class name (e.g., 'Gpt2')
            full_model_id: Full identifier of the HuggingFace model
                (e.g., 'openai-community/gpt2-large')
            task_type: The task type this model will perform (e.g., 'CausalLM')
            default_download_mode: Default download mode to use ('config_only' or 'full_model')
            default_artifact_dir: Default artifact directory to use

        Returns:
            type: The created and registered model class
        """
        # Get the appropriate Auto Class for this task type
        model_class_auto = TASK_MAPPINGS.get(task_type, TASK_MAPPINGS["default"])["auto_class"]
        class_name = f"Auto{camel_name}For{task_type}"

        # If we've already created this class in a different process, return it
        if class_name in HuggingFaceModelManager._created_model_classes:
            logger.info(f"Using previously created class: {class_name}")
            return HuggingFaceModelManager._created_model_classes[class_name]

        # Define initialization function for the wrapper
        def init_func(self, config=None):
            nn.Module.__init__(self)
            # NotAnnotated.__init__(self)
            self.model_name = full_model_id

            # Initialize the model using its full model id and Auto Class
            print(config)
            self.model = HuggingFaceModelManager.initialize_model(
                model_id=full_model_id,
                model_class=model_class_auto,
                config=config,
                default_download_mode=default_download_mode,
                default_artifact_dir=default_artifact_dir,
            )

        # Define forward function for the wrapper
        def forward_func(self, input_ids, attention_mask=None, labels=None, **kwargs):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels if labels is not None else None,
                **kwargs,
            )
            return outputs

        # Create the class dynamically
        wrapper_class = type(
            class_name,
            (nn.Module,),
            {
                "__init__": init_func,
                "forward": forward_func,
                "__doc__": f"HuggingFace {full_model_id} model wrapper for {task_type}.",
                "model_id": full_model_id,
            },
        )

        # Try to register the class, handling the case where it's already registered
        try:
            # registered_class = register_model(wrapper_class)
            registered_class = wrapper_class
            logger.info(f"Registered new class: {class_name}")
            HuggingFaceModelManager._created_model_classes[class_name] = registered_class
            return registered_class
        except RuntimeError as e:
            if "attempt to re-define previously registered" in str(e):
                logger.info(f"Class {class_name} already registered, retrieving from registry")
                # Commented out as get_registry is not defined
                # registry = get_registry("models")
                # registered_class = registry[class_name]
                registered_class = wrapper_class
                HuggingFaceModelManager._created_model_classes[class_name] = registered_class
                return registered_class
            else:
                # Re-raise if it's a different error
                raise

    @staticmethod
    def initialize_model(
        model_id: str,
        model_class: type,
        config: dict[str, Any] | str | None = None,
        default_download_mode: str = "config_only",
        default_artifact_dir: str | None = None,
    ) -> nn.Module:
        """
        Initialize a HuggingFace model based on download mode.

        Args:
            model_id: The HuggingFace model ID
            model_class: The AutoModel class to use
            config: Configuration object or path containing task_config parameters
            default_download_mode: Default download mode if not specified in config
            default_artifact_dir: Default artifact directory if not specified in config

        Returns:
            model: The initialized model
        """
        # Set defaults and extract from config if available
        download_mode = default_download_mode
        artifact_dir = default_artifact_dir
        if config is not None and isinstance(config, dict) and "task_config" in config:
            task_config_dict = config["task_config"]
            download_mode = task_config_dict.get("download_mode", download_mode)
            artifact_dir = task_config_dict.get("artifact_directory", artifact_dir)
        logger.info(
            f"Initializing model {model_id} for class {model_class} "
            f"with download_mode={download_mode}, artifact_dir={artifact_dir}"
        )

        # Load model from config as default
        if download_mode == "config_only":
            logger.info(f"Loading {model_id} from config")
            model, success = HuggingFaceModelManager.Loader.load_model_from_config(
                model_class, model_id, artifact_dir
            )
            if success:
                return model
            logger.warning(f"Failed to load {model_id} from config, trying full model")

        # Fall back to full model download if config failed
        logger.info(f"Loading {model_id} from full model")
        model, success = HuggingFaceModelManager.Loader.load_model_from_pretrained(
            model_class, model_id
        )
        if success:
            return model

        # If we got here, all attempts failed
        raise ImportError(
            f"Failed to initialize model {model_id} for class {model_class} "
            f"with download_mode={download_mode}, artifact_dir={artifact_dir} "
            f"after trying all methods"
        )

    class Loader:
        """Utility class for loading and downloading HuggingFace models and configs."""

        @staticmethod
        def load_model_from_config(
            model_class: type, model_id: str, artifact_dir: str | None = None
        ) -> tuple[nn.Module | None, bool]:
            """
            Load a model from a configuration file.

            Args:
                model_class: The AutoModel class to use
                model_id: The HuggingFace model ID
                artifact_dir: Directory to save configs (runtime directory)

            Returns:
                tuple: (model, success)
            """
            try:
                # Store artifact_dir for later use
                if artifact_dir:
                    HuggingFaceModelManager._artifact_dir = artifact_dir

                # Get or download config
                config_path = HuggingFaceModelManager.Loader.get_or_download_model_config(
                    model_id, artifact_dir
                )

                if not config_path:
                    logger.warning(f"No config found locally for {model_id}")
                    return None, False

                # Ensure config_path has .json extension
                if not config_path.endswith(".json"):
                    config_path = f"{config_path}.json"

                logger.info(f"Initializing {model_id} from config at {config_path}")

                # Try with and without trust_remote_code
                try:
                    model_config = AutoConfig.from_pretrained(config_path)
                    # import pdb; pdb.set_trace()
                    # HARD-CODE num layers to 2
                    model_config.num_hidden_layers = 2
                    model = model_class.from_config(model_config)
                except ValueError as e:
                    if "trust_remote_code" in str(e):
                        model_config = AutoConfig.from_pretrained(
                            config_path, trust_remote_code=True
                        )
                        model = model_class.from_config(model_config, trust_remote_code=True)
                    else:
                        raise

                logger.info(f"Successfully initialized {model_id} from config at {config_path}")
                return model, True
            except Exception as e:
                logger.error(f"Error initializing {model_id} from config: {e}")
                return None, False

        @staticmethod
        def get_or_download_model_config(
            model_id: str, artifact_dir: str | None = None
        ) -> str | None:
            """
            Get or download a model config and copy to artifact dir if needed.

            Checks if config exists in modeling directory. If not, downloads to modeling directory.
            Once config is found or downloaded, copies to artifact directory for record keeping.

            Args:
                model_id: The HuggingFace model ID
                artifact_dir: Optional artifact directory for record keeping

            Returns:
                str: Path to the config file without .json extension, or None if not found
            """
            # Store artifact_dir for later use
            if artifact_dir:
                HuggingFaceModelManager._artifact_dir = artifact_dir

            # Get standard filenames and paths
            config_filename = HuggingFaceModelManager.Loader.get_config_filename(model_id)
            modeling_dir = HuggingFaceModelManager.Loader.get_hf_modeling_configs_dir()
            modeling_path = os.path.join(modeling_dir, config_filename)

            # If config exists in modeling directory, use it
            if os.path.exists(modeling_path):
                logger.info(f"Using existing config for {model_id}: {modeling_path}")

                # Copy to artifact directory if provided (for record keeping)
                if artifact_dir:
                    configs_dir = os.path.join(artifact_dir, "configs")
                    os.makedirs(configs_dir, exist_ok=True)
                    artifact_path = os.path.join(configs_dir, config_filename)

                    if not os.path.exists(artifact_path):
                        try:
                            with open(modeling_path) as src, open(artifact_path, "w") as dst:
                                dst.write(src.read())
                            logger.info(f"Copied config to artifact directory: {artifact_path}")
                        except Exception as e:
                            logger.warning(f"Failed to copy config to artifact directory: {e}")

                return modeling_path.replace(".json", "")

            # Config not found locally, try downloading
            logger.warning(
                f"Config for {model_id} not found locally. For best performance with parallel "
                f"execution, run download_hf_configs.py first. Attempting emergency download now..."
            )

            # Set up locking for parallel process safety
            lock_dir = os.path.join(modeling_dir, "locks")
            os.makedirs(lock_dir, exist_ok=True)
            lock_path = os.path.join(lock_dir, f"{model_id.replace('/', '_')}.lock")

            with FileLock(lock_path):
                # Check again after acquiring lock
                if os.path.exists(modeling_path):
                    return modeling_path.replace(".json", "")

                # Download with conservative rate limiting
                time.sleep(5.0 + random.uniform(0, 1.0))

                try:
                    # Try to download the config
                    logger.info(f"Emergency download of config for {model_id}")
                    try:
                        config = AutoConfig.from_pretrained(model_id)
                    except ValueError as e:
                        if "trust_remote_code" in str(e):
                            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                        else:
                            raise

                    config_json = config.to_json_string()

                    # Save to modeling directory
                    with open(modeling_path, "w") as f:
                        f.write(config_json)
                    logger.info(f"Saved config to: {modeling_path}")

                    # Save to artifact directory if specified (for record keeping)
                    if artifact_dir:
                        configs_dir = os.path.join(artifact_dir, "configs")
                        os.makedirs(configs_dir, exist_ok=True)
                        artifact_path = os.path.join(configs_dir, config_filename)

                        with open(artifact_path, "w") as f:
                            f.write(config_json)
                        logger.info(f"Also saved config to: {artifact_path}")

                    return modeling_path.replace(".json", "")

                except Exception as e:
                    logger.error(f"Failed to download config for {model_id}: {e}")
                    return None

        @staticmethod
        def get_config_filename(model_id: str) -> str:
            """
            Get standardized config filename for a model ID.

            Args:
                model_id: The HuggingFace model ID

            Returns:
                str: Standardized config filename with .json extension
            """
            safe_model_id = model_id.replace("/", "_")
            return f"{safe_model_id}_config.json"

        @staticmethod
        def get_hf_modeling_configs_dir() -> str:
            """
            Get the directory path for storing modeling configs based on package location.

            Directory structure::

                neuron_gspmd_tests/
                ├── modeling/
                │   ├── models/
                │   │   └── hf_model.py (this file)
                │   └── configs/
                │       └── huggingface/ (target directory)

            Returns:
                str: Path to the modeling configs directory
            """
            current_file = os.path.abspath(__file__)
            models_dir = os.path.dirname(current_file)
            modeling_dir = os.path.dirname(models_dir)
            hf_configs_dir = os.path.join(modeling_dir, "configs", "huggingface")
            os.makedirs(hf_configs_dir, exist_ok=True)
            return hf_configs_dir

        @staticmethod
        def load_model_from_pretrained(
            model_class: type, model_id: str
        ) -> tuple[nn.Module | None, bool]:
            """
            Load a pretrained model from HuggingFace.

            First tries to load from local cache. If not found, downloads the model
            and then loads from cache.

            Args:
                model_class: The HuggingFace AutoModel class to use
                model_id: The HuggingFace model ID

            Returns:
                tuple: (model, success)
            """
            try:
                # First try loading from local cache
                try:
                    model = model_class.from_pretrained(model_id, local_files_only=True)
                    logger.info(f"Loaded {model_id} from local cache")
                    return model, True
                except Exception as local_error:
                    logger.info(f"Model not in cache ({local_error}), trying to download")

                # If local load failed, download and then try again
                if HuggingFaceModelManager.Loader.download_model_pretrained(model_id):
                    model = model_class.from_pretrained(model_id, local_files_only=False)
                    logger.info(f"Successfully downloaded and loaded {model_id}")
                    return model, True
                else:
                    logger.error(f"Failed to download {model_id}")
                    return None, False
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {e}")
                return None, False

        @staticmethod
        def download_model_pretrained(model_id: str) -> bool:
            """
            Load full model files from HuggingFace cache.

            Uses file locks for parallel execution mode when multiple processes are downloading.
            Prioritizes loading from cache if this model has already been downloaded before.

            Args:
                model_id: The HuggingFace model ID

            Returns:
                bool: Whether the model files were downloaded
            """

            try:
                cache_dir = HuggingFaceModelManager.Loader.get_hf_model_cache_dir()
                lock_dir = os.path.join(os.path.dirname(cache_dir), "locks")
                os.makedirs(lock_dir, exist_ok=True)
                lock_path = os.path.join(lock_dir, f"{model_id.replace('/', '_')}.lock")

                with FileLock(lock_path):
                    try:
                        # Attempts to load the model files from cache first
                        snapshot_download(model_id, local_files_only=True)
                        logger.info(f"Model {model_id} already in cache")
                        return True
                    except Exception:
                        # Downloads the model files from HF to cache if none are found
                        logger.info(f"Downloading {model_id}...")
                        snapshot_download(model_id)
                        logger.info(f"Successfully downloaded {model_id}")
                        return True
            except Exception as e:
                logger.warning(f"Failed to download {model_id}: {e}")
                return False

        @staticmethod
        def get_hf_model_cache_dir() -> str:
            """
            Get the HuggingFace cache directory for full model downloads.

            By default, the huggingface cache is saved to "~/.cache/huggingface/hub".
            Set the environment variable HF_HOME to override (recommended):

            export HF_HOME="shared/$USER/.cache/huggingface/hub"

            Returns:
                str: Path to the HuggingFace cache directory
            """
            # Try HuggingFace home environment variables first
            cache_dir = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
            if not cache_dir:
                logger.warning(
                    "No cache directory set. Defaulting to ~/.cache/huggingface/hub. "
                    "Set environment variable HF_HOME"
                )
                # Fallback to default cache directory
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            os.makedirs(cache_dir, exist_ok=True)
            return cache_dir


def run_fwd_bwd(config_path):
    sl = 1024
    import torch_neuronx

    print("config_path is", config_path)
    device = "neuron"
    input_ids = torch.randint(0, sl, (1, sl)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    labels = torch.randint(0, sl, (1, sl)).to(device)
    model_config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    # HARD-CODE num layers to 2
    model_config._attention_implementation = "eager"
    if hasattr(model_config, "num_hidden_layers") and model_config.num_hidden_layers > 1:
        model_config.num_hidden_layers = 2
    else:
        model_config.num_encoder_layers = 2
        model_config.num_decoder_layers = 2
    model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    model.to(device)

    out = model(input_ids, attention_mask=attention_mask, labels=labels)
    if out.loss is None:
        out.loss = out.logits.sum()
    out.loss.backward()
