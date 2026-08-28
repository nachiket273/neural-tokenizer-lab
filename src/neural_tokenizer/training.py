import time

import torch.nn.functional as F


def train(model, loader, optimizer, device, steps: int):
    model.train()

    iterator = iter(loader)

    history = []

    for step in range(steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x, y = x.to(device), y.to(device)

        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        loss.backward()

        optimizer.step()

        elapsed = time.perf_counter() - start

        history.append(
            {
                "step": step,
                "loss": loss.item(),
                "time": elapsed,
            }
        )

        if step % 100 == 0:
            print(f"step={step:05d} " f"loss={loss.item():.4f} " f"time={elapsed:.3f}s")

    return history
