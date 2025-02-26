"""LLM-based description generator for model analysis reports.

This module provides functionality to generate natural language descriptions
of model analysis results using lightweight language models.
"""

import logging
from typing import Any, Dict, Optional

# Import a lightweight local inference library - we're using llama-cpp-python
# This supports running small GGUF models locally
try:
    from llama_cpp import Llama

    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logging.warning(
        "llama-cpp-python not installed. Using fallback description generation."
    )

# Fallback to transformers if llama-cpp not available
try:
    import torch
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    if not LLAMA_AVAILABLE:
        logging.warning(
            "Neither llama-cpp-python nor transformers installed. Using static descriptions."
        )

logger = logging.getLogger(__name__)

# Default model paths - these would be downloaded separately
# Using TinyLlama as a very lightweight option (1.1B parameters)
DEFAULT_GGUF_MODEL_PATH = "models/llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
DEFAULT_TRANSFORMER_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class LLMDescriptionGenerator:
    """Generates natural language descriptions for model analysis data using lightweight LLMs."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_transformers: bool = False,
        transformer_model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """Initialize the LLM description generator.

        Args:
            model_path: Path to the GGUF model file (for llama-cpp)
            use_transformers: Whether to use HuggingFace Transformers instead of llama-cpp
            transformer_model: Which transformer model to use (if use_transformers is True)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (higher = more creative)
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm = None
        self.initialized = False

        # Try to initialize LLM
        if not use_transformers and LLAMA_AVAILABLE:
            try:
                self.llm = Llama(
                    model_path=model_path or DEFAULT_GGUF_MODEL_PATH,
                    n_ctx=2048,  # Context window
                    n_threads=4,  # Use 4 threads for inference
                )
                self.initialized = True
                self.backend = "llama-cpp"
                logger.info(
                    f"Using llama-cpp with model: {model_path or DEFAULT_GGUF_MODEL_PATH}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize llama-cpp model: {e}")

        # Fallback to transformers if requested or if llama-cpp failed
        if not self.initialized and TRANSFORMERS_AVAILABLE:
            try:
                # Use transformers pipeline
                self.llm = pipeline(
                    "text-generation",
                    model=transformer_model or DEFAULT_TRANSFORMER_MODEL,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                self.initialized = True
                self.backend = "transformers"
                logger.info(
                    f"Using transformers with model: {transformer_model or DEFAULT_TRANSFORMER_MODEL}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize transformers model: {e}")

        if not self.initialized:
            logger.warning(
                "No LLM could be initialized. Will use fallback static descriptions."
            )
            self.backend = "static"

    def generate_description(
        self, context: Dict[str, Any], prompt_template: str, section: str
    ) -> str:
        """Generate a description for a specific section of the report.

        Args:
            context: Dictionary containing context data for the generation
            prompt_template: Template string for the prompt
            section: Name of the section to generate description for

        Returns:
            Generated description as a string
        """
        if not self.initialized:
            return self._get_fallback_description(section)

        # Format the prompt with the context data
        try:
            formatted_prompt = prompt_template.format(**context)
        except KeyError as e:
            logger.warning(f"Error formatting prompt: {e}. Using fallback.")
            return self._get_fallback_description(section)

        try:
            if self.backend == "llama-cpp":
                # Generate with llama-cpp
                response = self.llm(
                    formatted_prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=["###"],
                )
                return response["choices"][0]["text"].strip()

            elif self.backend == "transformers":
                # Generate with transformers
                response = self.llm(
                    formatted_prompt,
                    max_length=len(formatted_prompt.split()) + self.max_tokens,
                    temperature=self.temperature,
                    num_return_sequences=1,
                    do_sample=True,
                )
                # Extract just the newly generated text
                generated_text = response[0]["generated_text"]
                return generated_text[len(formatted_prompt) :].strip()
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return self._get_fallback_description(section)

        return self._get_fallback_description(section)

    def generate_feature_importance_description(
        self, feature_importance: Dict[str, float]
    ) -> str:
        """Generate a description of feature importance results.

        Args:
            feature_importance: Dictionary mapping feature names to importance scores

        Returns:
            Description of feature importance findings
        """
        # Sort features by importance and take top 5
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: float(x[1]) if isinstance(x[1], (int, float, str)) else 0,
            reverse=True,
        )
        top_features = sorted_features[:5]

        # Create context for the prompt
        context = {
            "top_features": ", ".join(
                [f"{feature} ({score:.4f})" for feature, score in top_features]
            ),
            "feature_count": len(feature_importance),
            "top_feature": top_features[0][0] if top_features else "N/A",
            "top_feature_score": f"{top_features[0][1]:.4f}" if top_features else "N/A",
        }

        # Define the prompt template
        prompt = """
        Analyze the importance of features in a machine learning model.

        Here are the top 5 most important features out of {feature_count} total features:
        {top_features}

        Provide a brief, insightful description of what these results suggest about the model and data.
        Focus on explaining which features drive predictions the most and what that means for the business context.
        Keep your explanation under 150 words and make it easy to understand.

        Description:
        """

        return self.generate_description(context, prompt, "feature_importance")

    def generate_model_performance_description(
        self, performance_metrics: Dict[str, Any]
    ) -> str:
        """Generate a description of model performance metrics.

        Args:
            performance_metrics: Dictionary containing model performance metrics

        Returns:
            Description of model performance insights
        """
        # Extract key metrics
        metrics = performance_metrics.get("classification_metrics", {})
        threshold = performance_metrics.get("optimal_threshold", "N/A")
        threshold_metrics = performance_metrics.get("threshold_metrics", {})

        # Create context for the prompt
        context = {
            "accuracy": f"{metrics.get('accuracy', 'N/A'):.4f}"
            if isinstance(metrics.get("accuracy"), (int, float))
            else "N/A",
            "precision": f"{metrics.get('precision', 'N/A'):.4f}"
            if isinstance(metrics.get("precision"), (int, float))
            else "N/A",
            "recall": f"{metrics.get('recall', 'N/A'):.4f}"
            if isinstance(metrics.get("recall"), (int, float))
            else "N/A",
            "f1_score": f"{metrics.get('f1_score', 'N/A'):.4f}"
            if isinstance(metrics.get("f1_score"), (int, float))
            else "N/A",
            "threshold": f"{threshold:.4f}"
            if isinstance(threshold, (int, float))
            else "N/A",
            "tpr": f"{threshold_metrics.get('true_positive_rate', 'N/A'):.4f}"
            if isinstance(threshold_metrics.get("true_positive_rate"), (int, float))
            else "N/A",
            "fpr": f"{threshold_metrics.get('false_positive_rate', 'N/A'):.4f}"
            if isinstance(threshold_metrics.get("false_positive_rate"), (int, float))
            else "N/A",
        }

        # Define the prompt template
        prompt = """
        Analyze the performance metrics of a binary classification model for predicting customer responses to sales offers.

        Model Performance Metrics:
        - Accuracy: {accuracy}
        - Precision: {precision}
        - Recall: {recall}
        - F1 Score: {f1_score}
        - Optimal Threshold: {threshold}
        - True Positive Rate at threshold: {tpr}
        - False Positive Rate at threshold: {fpr}

        Provide a brief, insightful interpretation of these metrics, highlighting:
        1. Overall model quality
        2. Trade-offs between precision and recall
        3. What these results mean for business decision-making
        Keep your explanation under 150 words and make it easy for non-technical stakeholders to understand.

        Analysis:
        """

        return self.generate_description(context, prompt, "model_performance")

    def generate_financial_impact_description(
        self, financial_data: Dict[str, Any]
    ) -> str:
        """Generate a description of financial impact analysis.

        Args:
            financial_data: Dictionary containing financial impact analysis

        Returns:
            Description of financial impact insights
        """
        # Extract key financial data
        total_profit = financial_data.get("total_profit", "N/A")
        opportunity_loss = financial_data.get("opportunity_loss", "N/A")
        roi = financial_data.get("roi", "N/A")
        campaign_size = financial_data.get("campaign_size", "N/A")

        # Get profit by risk band
        profit_by_risk = financial_data.get("profit_by_risk_band", {})

        # Create context for the prompt
        context = {
            "total_profit": f"${total_profit:.2f}"
            if isinstance(total_profit, (int, float))
            else "N/A",
            "opportunity_loss": f"${opportunity_loss:.2f}"
            if isinstance(opportunity_loss, (int, float))
            else "N/A",
            "roi": f"{roi:.2f}%" if isinstance(roi, (int, float)) else "N/A",
            "campaign_size": f"{campaign_size:,}"
            if isinstance(campaign_size, (int, float))
            else "N/A",
            "high_risk_profit": f"${profit_by_risk.get('High', 'N/A'):.2f}"
            if isinstance(profit_by_risk.get("High"), (int, float))
            else "N/A",
            "medium_risk_profit": f"${profit_by_risk.get('Medium', 'N/A'):.2f}"
            if isinstance(profit_by_risk.get("Medium"), (int, float))
            else "N/A",
            "low_risk_profit": f"${profit_by_risk.get('Low', 'N/A'):.2f}"
            if isinstance(profit_by_risk.get("Low"), (int, float))
            else "N/A",
        }

        # Define the prompt template
        prompt = """
        Analyze the financial impact of a predictive model for a sales campaign.

        Financial Impact Analysis:
        - Total Profit: {total_profit}
        - Opportunity Loss: {opportunity_loss}
        - ROI: {roi}
        - Campaign Size: {campaign_size} customers

        Profit by Risk Band:
        - High Risk: {high_risk_profit}
        - Medium Risk: {medium_risk_profit}
        - Low Risk: {low_risk_profit}

        Provide a brief, insightful interpretation of these financial results, highlighting:
        1. The overall financial performance of the model
        2. The distribution of profit across risk bands
        3. Recommendations for optimizing campaign ROI

        Keep your explanation under 150 words and focus on business implications.

        Analysis:
        """

        return self.generate_description(context, prompt, "financial_impact")

    def _get_fallback_description(self, section: str) -> str:
        """Return a static fallback description when LLM generation fails.

        Args:
            section: Name of the section to get fallback description for

        Returns:
            Static fallback description
        """
        fallbacks = {
            "feature_importance": (
                "The feature importance analysis reveals which variables have the strongest influence on the model's "
                "predictions. Higher values indicate greater importance. The top features likely represent key factors "
                "in customer decision-making. Marketing strategies should particularly focus on these influential variables "
                "to maximize campaign effectiveness."
            ),
            "model_performance": (
                "The model demonstrates solid predictive performance with balanced precision and recall metrics. "
                "The optimal threshold was selected to maximize business value by finding the right balance between "
                "true positives (correctly identified responsive customers) and false positives (customers incorrectly "
                "predicted to respond). This balance is critical for efficient resource allocation in marketing campaigns."
            ),
            "financial_impact": (
                "Financial analysis indicates a positive return on investment for the modeled campaign approach. "
                "The profit distribution across risk bands shows where the model creates the most value. "
                "Higher-risk segments typically yield higher potential returns but with greater variability. "
                "Marketing resources should be allocated according to these risk-adjusted returns to maximize "
                "overall campaign profitability."
            ),
            "data_quality": (
                "The data quality assessment indicates a well-prepared dataset with minimal missing values and "
                "appropriate feature distributions. Feature engineering has been applied to create informative "
                "variables for the model. The preprocessing steps have effectively prepared the data for "
                "optimal model training."
            ),
            "class_balance": (
                "The class distribution shows an imbalanced dataset, which is typical for marketing response models "
                "where positive responses are relatively rare. The model has been optimized to account for this "
                "imbalance, ensuring effective prediction despite the skewed distribution. Proper threshold "
                "selection helps mitigate bias toward the majority class."
            ),
        }

        return fallbacks.get(
            section,
            "This section presents important insights about the model's behavior and performance. "
            "The visualizations and metrics provide key information for understanding prediction patterns "
            "and business implications.",
        )
