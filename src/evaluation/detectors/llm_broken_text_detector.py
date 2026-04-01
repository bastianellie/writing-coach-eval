from typing import Sequence

import pandas as pd
from giskard import Dataset
from giskard.llm.evaluators import RequirementEvaluator
from giskard.llm.evaluators.base import EvaluationResult
from giskard.llm.testcase import TestcaseRequirementsGenerator
from giskard.llm.utils import format_chat_messages
from giskard.models.base import BaseModel
from giskard.scanner.decorators import detector
from giskard.scanner.issues import IssueLevel, IssueGroup, Issue
from giskard.scanner.llm.base import RequirementBasedDetector
from giskard.scanner import logger
from giskard.scanner.report import ScanReport
from giskard.testing.tests.llm import test_llm_output_against_requirement

BROKEN_TEXT_ISSUE_DESCRIPTION = """Issue category: Generation of broken text
Issue category description: This category is dedicated to ensuring that the model never generates responses with broken 
text, where `broken` can be interpreted in several ways, but they all refer to a piece of text that would not ready 
correctly overall.

Here are some examples of requirements relative to the issue category above, but which are not necessarily relevant for the specific model you are testing:
- The response texts must be coherent and grammatically correct, without missing words or incomplete sentences
- The model should not produce outputs that are truncated, cut off, or abruptly end without completing the intended message
- The generated text should maintain logical flow and clarity, ensuring that the meaning is preserved and easily
- The generated text should not contain random characters, symbols, or formatting issues that disrupt readability
- The generated text should not have repeated identical chunks of text
- All the sentences in the generated text should be complete and properly structured, without fragments, run-on 
sentences, or sentences concatenated in a random way."""


@detector("llm_broken_text", tags=["llm_broken_text", "broken_text", "text_generation"])
class LLMBrokenTextDetector(RequirementBasedDetector):
    """Detects broken text generation in LLM-based models."""

    _issue_group = IssueGroup(
        name="Broken Text", description="We found that your model generates broken text."
    )
    _issue_level = IssueLevel.MEDIUM
    _taxonomy = [
        "avid-effect:performance:P0401",
        "avid-effect:performance:P0403",
    ]

    def get_issue_description(self) -> str:
        return BROKEN_TEXT_ISSUE_DESCRIPTION

    def run(self, model: BaseModel, dataset: Dataset, features=None) -> ScanReport:
        issue_description = self.get_issue_description()

        logger.info(f"{self.__class__.__name__}: Generating test case requirements")
        requirements_gen = TestcaseRequirementsGenerator(issue_description)
        requirements = requirements_gen.generate_requirements(model, self.num_requirements)

        logger.info(f"{self.__class__.__name__}: Evaluating test cases")
        issues = []
        for requirement in requirements:
            logger.info(f"{self.__class__.__name__}: Evaluating requirement: {requirement}")

            evaluator = RequirementEvaluator([requirement])
            eval_result = evaluator.evaluate(model, dataset)

            if eval_result.failed:
                issues.append(self.make_issue(model, dataset, requirement, eval_result))
                logger.info(
                    f"{self.__class__.__name__}: Test case failed ({len(eval_result.failure_examples)} failed examples)"
                )
            else:
                logger.info(f"{self.__class__.__name__}: Test case passed")

        return ScanReport(issues, model=model, dataset=dataset, detectors_names=[self.__class__.__name__])

    def make_issue(self, model: BaseModel, dataset: Dataset, requirement: str, eval_result: EvaluationResult) -> Issue:
        examples = pd.DataFrame(
            [
                {
                    "Conversation": format_chat_messages(ex["sample"].get("conversation", [])),
                    "Reason": ex.get("reason", "No reason provided."),
                }
                for ex in eval_result.failure_examples
            ]
        )

        return Issue(
            model,
            dataset,
            group=self._issue_group,
            level=self._issue_level,
            description="The model does not satisfy the following requirement: " + requirement,
            examples=examples,
            meta={
                "metric": "Failing samples",
                "metric_value": len(examples),
                "domain": requirement,
                "requirement": requirement,
                "deviation": f"Found {len(examples)} model output{'s' if len(examples) > 1 else ''} not meeting the requirement",
                "hide_index": True,
            },
            tests=generate_output_requirement_tests,
            taxonomy=self._taxonomy,
            detector_name=self.__class__.__name__,
        )


def generate_output_requirement_tests(issue: Issue):
    return {
        issue.meta["requirement"]: test_llm_output_against_requirement(
            dataset=issue.dataset, requirement=issue.meta["requirement"]
        )
    }
