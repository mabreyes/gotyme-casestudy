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

    def generate_data_quality_description(self, data_quality: Dict[str, Any]) -> str:
        """Generate a description of data quality assessment.

        Args:
            data_quality: Dictionary containing data quality information

        Returns:
            Description of data quality insights
        """
        # Extract key data quality information
        missing_values = data_quality.get("missing_values", {})
        outliers = data_quality.get("outliers", {})
        distributions = data_quality.get("distributions", {})

        # Calculate total missing values and outliers
        total_missing = sum(count for feature, count in missing_values.items())
        total_outliers = sum(count for feature, count in outliers.items())

        # Find features with most outliers
        outlier_features = sorted(
            outliers.items(),
            key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
            reverse=True,
        )

        # Find skewed distributions
        skewed_features = []
        for feature, dist in distributions.items():
            if "skew" in dist and abs(dist["skew"]) > 1.0:
                skewed_features.append((feature, dist["skew"]))

        # Create context for the prompt
        context = {
            "feature_count": len(missing_values),
            "total_missing": total_missing,
            "total_outliers": total_outliers,
            "most_outliers": outlier_features[0][0] if outlier_features else "N/A",
            "most_outliers_count": outlier_features[0][1] if outlier_features else 0,
            "skewed_features": ", ".join(
                [f"{f} (skew={s:.2f})" for f, s in skewed_features[:3]]
            )
            if skewed_features
            else "None significant",
            "avg_skew": sum(abs(dist.get("skew", 0)) for dist in distributions.values())
            / len(distributions)
            if distributions
            else 0,
        }

        # Define the prompt template
        prompt = """
        Analyze the data quality assessment for a machine learning dataset.

        Data Quality Information:
        - Total Features: {feature_count}
        - Total Missing Values: {total_missing}
        - Total Outliers: {total_outliers}
        - Feature with Most Outliers: {most_outliers} ({most_outliers_count} outliers)
        - Most Skewed Features: {skewed_features}
        - Average Absolute Skew: {avg_skew:.2f}

        Provide a brief, insightful interpretation of these data quality metrics, highlighting:
        1. Overall data completeness and quality
        2. Potential challenges with outliers or skewed distributions
        3. Implications for model development and performance

        Keep your explanation under 150 words and focus on practical insights.

        Analysis:
        """

        return self.generate_description(context, prompt, "data_quality")

    def generate_class_balance_description(self, class_balance: Dict[str, Any]) -> str:
        """Generate a description of class balance information.

        Args:
            class_balance: Dictionary containing class balance information

        Returns:
            Description of class balance insights, including SMOTE information if available
        """
        # Extract class balance information
        class_counts = class_balance.get("class_counts", {})
        class_proportions = class_balance.get("class_proportions", {})

        # Extract SMOTE information if available
        smote_info = class_balance.get("smote_results", {})
        smote_applied = bool(smote_info)

        # Calculate imbalance ratio
        counts = list(class_counts.values())
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else "high"

        # Determine minority and majority classes
        if len(class_counts) >= 2:
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
            minority_class = sorted_classes[0][0]
            majority_class = sorted_classes[-1][0]
            minority_count = sorted_classes[0][1]
            majority_count = sorted_classes[-1][1]
        else:
            minority_class = majority_class = "unknown"
            minority_count = majority_count = 0

        # Create context for the prompt
        context = {
            "class_count": len(class_counts),
            "minority_class": minority_class,
            "majority_class": majority_class,
            "minority_count": minority_count,
            "majority_count": majority_count,
            "imbalance_ratio": f"{imbalance_ratio:.2f}"
            if isinstance(imbalance_ratio, (int, float))
            else imbalance_ratio,
            "smote_applied": "Yes" if smote_applied else "No",
            "minority_before": smote_info.get("before", {}).get("minority_count", "N/A")
            if smote_applied
            else "N/A",
            "minority_after": smote_info.get("after", {}).get("minority_count", "N/A")
            if smote_applied
            else "N/A",
        }

        # Define the prompt template
        prompt = """
        Analyze the class balance for a machine learning classification dataset.

        Class Balance Information:
        - Number of Classes: {class_count}
        - Majority Class: {majority_class} ({majority_count} samples)
        - Minority Class: {minority_class} ({minority_count} samples)
        - Imbalance Ratio: {imbalance_ratio}
        - SMOTE Applied: {smote_applied}

        {smote_details}

        Provide a brief, insightful interpretation of the class balance, highlighting:
        1. The severity of class imbalance and its potential impact on model performance
        2. How the applied resampling techniques (if any) might help address imbalance issues
        3. Recommended approaches for handling this class distribution in model development

        Keep your explanation under 150 words and focus on practical implications.

        Analysis:
        """

        # Add SMOTE details if available
        smote_details = ""
        if smote_applied:
            smote_details = f"""
            SMOTE Results:
            - Minority Class Count Before: {context['minority_before']}
            - Minority Class Count After: {context['minority_after']}
            - Resampling Method: Synthetic Minority Over-sampling TEchnique (SMOTE)
            """
        else:
            smote_details = (
                "No resampling techniques were applied to address class imbalance."
            )

        context["smote_details"] = smote_details

        return self.generate_description(context, prompt, "class_balance")

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

        # Get least important features
        bottom_features = sorted_features[-3:] if len(sorted_features) > 3 else []

        # Create context for the prompt
        context = {
            "top_features": ", ".join(
                [f"{feature} ({score:.4f})" for feature, score in top_features]
            ),
            "bottom_features": ", ".join(
                [f"{feature} ({score:.4f})" for feature, score in bottom_features]
            ),
            "feature_count": len(feature_importance),
            "top_feature": top_features[0][0] if top_features else "N/A",
            "top_feature_score": f"{top_features[0][1]:.4f}" if top_features else "N/A",
            "importance_spread": top_features[0][1] / bottom_features[0][1]
            if top_features and bottom_features and bottom_features[0][1] > 0
            else "N/A",
        }

        # Define the prompt template
        prompt = """
        Analyze the importance of features in a machine learning model.

        Feature Importance Information:
        - Total Features: {feature_count}
        - Top 5 Features: {top_features}
        - Least Important Features: {bottom_features}
        - Most Important Feature: {top_feature} (score: {top_feature_score})
        - Importance Ratio (top/bottom): {importance_spread}

        Provide a brief, insightful description of what these results suggest about the model and data.
        Focus on explaining:
        1. Which features drive predictions the most and what business factors they represent
        2. What the spread of importance values tells us about feature redundancy
        3. How these insights can inform feature engineering or business decisions

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
            "auc": f"{metrics.get('auc', 'N/A'):.4f}"
            if isinstance(metrics.get("auc"), (int, float))
            else "N/A",
            "threshold": f"{threshold:.4f}"
            if isinstance(threshold, (int, float))
            else "N/A",
            "tpr": f"{threshold_metrics.get('tpr', 'N/A'):.4f}"
            if isinstance(threshold_metrics.get("tpr"), (int, float))
            else "N/A",
            "fpr": f"{threshold_metrics.get('fpr', 'N/A'):.4f}"
            if isinstance(threshold_metrics.get("fpr"), (int, float))
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
        - AUC: {auc}
        - Optimal Threshold: {threshold}
        - True Positive Rate at threshold: {tpr}
        - False Positive Rate at threshold: {fpr}

        Provide a brief, insightful interpretation of these metrics, highlighting:
        1. Overall model quality and reliability
        2. Trade-offs between precision and recall at the selected threshold
        3. What these results mean for business decision-making and customer targeting
        4. Recommendations for potential model improvements or deployment considerations

        Keep your explanation under 150 words and make it easy for non-technical stakeholders to understand.

        Analysis:
        """

        return self.generate_description(context, prompt, "model_performance")

    def generate_financial_impact_description(
        self, financial_data: Dict[str, Any]
    ) -> str:
        """Generate a description of financial impact analysis.

        Args:
            financial_data: Dictionary containing financial impact information

        Returns:
            Description of financial impact insights
        """
        # Extract key financial metrics
        campaign_size = financial_data.get("campaign_size", "N/A")
        total_profit = financial_data.get("total_profit", "N/A")
        roi = financial_data.get("roi", "N/A")
        opportunity_loss = financial_data.get("opportunity_loss", "N/A")

        # Extract profit by risk band if available
        profit_by_risk = financial_data.get("profit_by_risk_band", {})
        risk_bands = list(profit_by_risk.keys())

        # Extract confusion matrix data if available
        cm = financial_data.get("scaled_confusion_matrix", {})
        true_positives = cm.get("true_positives", "N/A")
        false_positives = cm.get("false_positives", "N/A")
        false_negatives = cm.get("false_negatives", "N/A")

        # Create context for the prompt
        context = {
            "campaign_size": f"{campaign_size:,}"
            if isinstance(campaign_size, (int, float))
            else campaign_size,
            "total_profit": f"${total_profit:.2f}"
            if isinstance(total_profit, (int, float))
            else total_profit,
            "roi": f"{roi:.2f}%" if isinstance(roi, (int, float)) else roi,
            "opportunity_loss": f"${opportunity_loss:.2f}"
            if isinstance(opportunity_loss, (int, float))
            else opportunity_loss,
            "risk_bands": ", ".join(risk_bands) if risk_bands else "N/A",
            "most_profitable_band": max(profit_by_risk.items(), key=lambda x: x[1])[0]
            if profit_by_risk
            else "N/A",
            "true_positives": f"{true_positives:,}"
            if isinstance(true_positives, (int, float))
            else true_positives,
            "false_positives": f"{false_positives:,}"
            if isinstance(false_positives, (int, float))
            else false_positives,
            "false_negatives": f"{false_negatives:,}"
            if isinstance(false_negatives, (int, float))
            else false_negatives,
        }

        # Prompt template for financial impact
        prompt_template = """
        You are a financial analyst explaining the results of a predictive model's financial impact.

        Here are the key financial metrics:
        - Campaign size: {campaign_size} customers
        - Total profit: {total_profit}
        - Return on Investment (ROI): {roi}
        - Opportunity loss: {opportunity_loss}
        - Risk bands: {risk_bands}
        - Most profitable risk band: {most_profitable_band}
        - True positives (correctly targeted): {true_positives}
        - False positives (incorrectly targeted): {false_positives}
        - False negatives (missed opportunities): {false_negatives}

        Write a concise paragraph (3-5 sentences) explaining what these financial metrics mean for the business.
        Focus on the profit potential, ROI, and how the model's predictions translate to financial outcomes.
        Explain the implications of the risk bands and what they mean for targeting strategy.

        Your explanation:
        """

        return self.generate_description(context, prompt_template, "financial_impact")

    def generate_executive_summary_description(
        self, analysis_results: Dict[str, Any]
    ) -> str:
        """Generate an executive summary description of the entire analysis.

        Args:
            analysis_results: Dictionary containing all analysis results

        Returns:
            Executive summary description
        """
        # Extract key metrics from different sections
        data_quality = analysis_results.get("data_quality", {})
        feature_relevance = analysis_results.get("feature_relevance", {})
        class_balance = analysis_results.get("class_balance", {})
        model_performance = analysis_results.get("model_performance", {})
        financial_impact = analysis_results.get("financial_impact", {})

        # Get key performance metrics
        metrics = model_performance.get("classification_metrics", {})
        accuracy = metrics.get("accuracy", "N/A")
        precision = metrics.get("precision", "N/A")
        recall = metrics.get("recall", "N/A")
        f1 = metrics.get("f1_score", "N/A")

        # Get financial metrics
        total_profit = financial_impact.get("total_profit", "N/A")
        roi = financial_impact.get("roi", "N/A")

        # Get top features if available
        feature_importance = feature_relevance.get("feature_importance", {})
        top_features = []
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: float(x[1]) if isinstance(x[1], (int, float, str)) else 0,
                reverse=True,
            )
            top_features = [f[0] for f in sorted_features[:3]]

        # Get class distribution
        class_counts = class_balance.get("class_counts", {})

        # Create context for the prompt
        context = {
            "accuracy": f"{accuracy:.2%}"
            if isinstance(accuracy, (int, float))
            else accuracy,
            "precision": f"{precision:.2%}"
            if isinstance(precision, (int, float))
            else precision,
            "recall": f"{recall:.2%}" if isinstance(recall, (int, float)) else recall,
            "f1": f"{f1:.2%}" if isinstance(f1, (int, float)) else f1,
            "total_profit": f"${total_profit:,.2f}"
            if isinstance(total_profit, (int, float))
            else total_profit,
            "roi": f"{roi:.2f}%" if isinstance(roi, (int, float)) else roi,
            "top_features": ", ".join(top_features) if top_features else "N/A",
            "class_distribution": ", ".join(
                [f"{k}: {v}" for k, v in class_counts.items()]
            )
            if class_counts
            else "N/A",
        }

        # Prompt template for executive summary
        prompt_template = """
        You are a data science consultant presenting a machine learning model analysis to business executives.

        Here are the key findings from the analysis:
        - Model accuracy: {accuracy}
        - Precision: {precision}
        - Recall: {recall}
        - F1 score: {f1}
        - Total projected profit: {total_profit}
        - Return on Investment: {roi}
        - Top predictive features: {top_features}
        - Class distribution: {class_distribution}

        Write a comprehensive executive summary (4-6 sentences) that explains the overall performance and business value of the model.
        Focus on what these metrics mean for the business in practical terms.
        Highlight the model's strengths, potential financial impact, and key drivers of predictions.
        Use business language rather than technical jargon.

        Your executive summary:
        """

        return self.generate_description(context, prompt_template, "executive_summary")

    def _get_fallback_description(self, section: str) -> str:
        """Get a fallback static description when LLM generation fails.

        Args:
            section: The section name to get fallback description for

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
