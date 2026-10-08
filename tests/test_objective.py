"""Exercise the production loss with tiny tensors, without loading a checkpoint.

These tests require PyTorch. AST extraction avoids importing the large-model
data stack solely to run the actual compute_loss function on CPU.
"""

import ast
import importlib.util
import types
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install PyTorch to check loss gradients")
class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch.nn.functional as F
        cls.torch = torch
        source = Path(__file__).resolve().parents[1] / "scripts" / "train_nft_sqlite.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        clamp = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "straight_through_clamp_min")
        trainer = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "NFTSQLiteTrainer")
        loss = next(node for node in trainer.body
                    if isinstance(node, ast.FunctionDef) and node.name == "compute_loss")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, clamp, loss], type_ignores=[]))
        namespace = {"torch": torch, "F": F}
        exec(compile(module, str(source), "exec"), namespace)
        cls.loss_function = staticmethod(namespace["compute_loss"])

    def loss_and_gradient(self, reward, weight):
        torch = self.torch
        logits = torch.tensor([[[0.2, -0.1, 0.3], [0.1, 0.4, -0.2], [0.3, 0.1, 0.0]]],
                              dtype=torch.float32, requires_grad=True)
        inputs = {"input_ids": torch.tensor([[0, 1, 2]]),
                  "old_logprobs": torch.tensor([[0.0, -1.5, -1.2]]),
                  "response_mask": torch.tensor([[0.0, 1.0, 1.0]]),
                  "reward": torch.tensor([float(reward)]), "r_hat": torch.tensor([0.5])}
        if weight is not None:
            inputs["evidence_weight"] = torch.tensor([weight])
        arguments = types.SimpleNamespace(nft_log_ratio_max=10, nft_min_r_hat=0.02,
                                           nft_max_r_hat=0.98, nft_clamp_positive=0,
                                           nft_clamp_negative=1, nft_neg_weight=1,
                                           nft_normalize=1, nft_loss_scale=1,
                                           nft_log_components=False)
        trainer = types.SimpleNamespace(args=arguments)
        model = lambda **kwargs: types.SimpleNamespace(logits=logits)
        loss = self.loss_function(trainer, model, inputs)
        loss.backward()
        return loss.detach(), logits.grad

    def test_unit_weight_recovers_nft(self):
        for reward in (0, 1):
            original_loss, original_gradient = self.loss_and_gradient(reward, None)
            loss, gradient = self.loss_and_gradient(reward, 1.0)
            self.torch.testing.assert_close(loss, original_loss)
            self.torch.testing.assert_close(gradient, original_gradient)

    def test_positive_weight_changes_only_positive_gradient(self):
        _, positive = self.loss_and_gradient(1, 1.0)
        _, weighted_positive = self.loss_and_gradient(1, 1.1)
        self.torch.testing.assert_close(weighted_positive, positive * 1.1)
        _, negative = self.loss_and_gradient(0, 1.0)
        _, weighted_negative = self.loss_and_gradient(0, 1.1)
        self.torch.testing.assert_close(weighted_negative, negative)
        self.assertGreater(negative.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
