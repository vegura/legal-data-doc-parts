import logging
from numbers import Real

import networkx as nx
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _distance_mtrx(
    embeddings_1,
    embeddings_2,
    global_batch_size,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    embeddings_1 = torch.as_tensor(embeddings_1, dtype=torch.float32, device=device)
    embeddings_2 = torch.as_tensor(embeddings_2, dtype=torch.float32, device=device)

    n1 = embeddings_1.shape[0]
    n2 = embeddings_2.shape[0]

    similarities = torch.empty((n1, n2), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for global_start_1 in range(0, n1, global_batch_size):
            global_end_1 = min(global_start_1 + global_batch_size, n1)
            x = F.normalize(
                embeddings_1[global_start_1:global_end_1],
                p=2,
                dim=1,
            )

            for global_start_2 in range(0, n2, global_batch_size):
                global_end_2 = min(global_start_2 + global_batch_size, n2)
                y = F.normalize(
                    embeddings_2[global_start_2:global_end_2],
                    p=2,
                    dim=1,
                )

                similarities[
                    global_start_1:global_end_1,
                    global_start_2:global_end_2,
                ] = x @ y.T

        return similarities


def _transition_matrix(A: torch.Tensor) -> torch.Tensor:
    degree = A.sum(dim=1)
    safe_degree = torch.where(degree == 0, torch.ones_like(degree), degree)
    return A / safe_degree[:, None]


def _validate_graph(
    graph: nx.Graph,
    graph_name: str,
    device: torch.device,
) -> torch.Tensor:
    """Validate an attributed graph and return its embedding matrices."""
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{graph_name} must be a networkx.Graph, got {type(graph).__name__}")

    def collect_embeddings(items, attribute_name, item_name):
        embeddings = []
        expected_dimension = None

        for identifier, attributes in items:
            if attribute_name not in attributes:
                raise ValueError(
                    f"{graph_name} {item_name} {identifier!r} is missing "
                    f"the {attribute_name!r} attribute"
                )

            try:
                embedding = torch.as_tensor(
                    attributes[attribute_name],
                    dtype=torch.float32,
                    device=device,
                )
            except (TypeError, ValueError, RuntimeError, OverflowError) as error:
                raise TypeError(
                    f"{graph_name} {item_name} {identifier!r} has a non-numeric "
                    f"{attribute_name!r} attribute"
                ) from error

            if embedding.ndim != 1 or embedding.numel() == 0:
                raise ValueError(
                    f"{graph_name} {item_name} {identifier!r} must have a non-empty, "
                    f"one-dimensional {attribute_name!r} attribute"
                )
            if not torch.isfinite(embedding).all():
                raise ValueError(
                    f"{graph_name} {item_name} {identifier!r} has non-finite values "
                    f"in its {attribute_name!r} attribute"
                )

            if expected_dimension is None:
                expected_dimension = embedding.numel()
            elif embedding.numel() != expected_dimension:
                raise ValueError(
                    f"{graph_name} {attribute_name!r} attributes must all have the "
                    f"same dimension; expected {expected_dimension}, got "
                    f"{embedding.numel()} for {item_name} {identifier!r}"
                )

            embeddings.append(embedding)

        if not embeddings:
            raise ValueError(f"{graph_name} must contain at least one {item_name}")

        return torch.stack(embeddings)

    node_embeddings = collect_embeddings(
        graph.nodes(data=True),
        "embedding",
        "node",
    )
    return node_embeddings


def _validate_inputs(G1, G2, alpha, tol, mask, device: str | torch.device):
    """Validate the public API inputs and return device-resident tensors."""
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be a string or torch.device")
    try:
        device = torch.device(device)
    except RuntimeError as error:
        raise ValueError(f"invalid device {device!r}") from error
    try:
        torch.empty(0, device=device)
    except (RuntimeError, NotImplementedError) as error:
        raise RuntimeError(f"PyTorch device {device} is not available") from error

    emb1 = _validate_graph(G1, "G1", device)
    emb2 = _validate_graph(G2, "G2", device)

    if emb1.shape[1] != emb2.shape[1]:
        raise ValueError(
            "G1 and G2 node embeddings must have the same dimension; "
            f"got {emb1.shape[1]} and {emb2.shape[1]}"
        )

    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a real number")
    alpha = float(alpha)
    if not torch.isfinite(torch.tensor(alpha)) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and between 0 and 1")

    if isinstance(tol, bool) or not isinstance(tol, Real):
        raise TypeError("tol must be a real number")
    tol = float(tol)
    if not torch.isfinite(torch.tensor(tol)) or tol <= 0:
        raise ValueError("tol must be finite and greater than 0")

    if mask is not None:
        try:
            mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
        except (TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise TypeError("mask must contain numeric values") from error

        if not torch.isfinite(mask).all():
            raise ValueError("mask must contain only finite values")
        try:
            torch.broadcast_to(mask, (len(G1), len(G2)))
        except RuntimeError as error:
            raise ValueError(
                f"mask must be broadcastable to {(len(G1), len(G2))}; got {mask.shape}"
            ) from error

    return emb1, emb2, alpha, tol, mask, device


@torch.inference_mode()
def cross_graph_similarity(
    G1: nx.Graph,
    G2: nx.Graph,
    alpha,
    tol=1e-6,
    mask=None,
    device: str | torch.device = "cuda",
    unbias_step: bool = False,
) -> torch.Tensor:
    """Estimate node correspondences between two attributed graphs.

    The initial score for each node pair is the cosine similarity between
    their ``"embedding"`` attributes. The function then repeatedly propagates
    those scores through the row-normalized adjacency matrices of both graphs
    and blends the propagated scores with the initial similarities::

        S_new = alpha * (P1.T @ S @ P2) + (1 - alpha) * S_0

    The propagated term is divided by its positive maximum before blending.
    Iteration stops when the Frobenius norm of the score change is below
    ``tol``, or after 100 iterations. Rows and columns in the result follow
    the iteration order of ``G1.nodes`` and ``G2.nodes``, respectively.

    :param G1: First graph, containing at least one node and one edge. Every
        node must provide a numeric ``"embedding"`` attribute, and every edge
        must provide a numeric ``"relation_embeddings"`` attribute.
    :param G2: Second graph, with the same attribute requirements as ``G1``.
    :param alpha: Weight assigned to structural propagation. Values in
        ``[0, 1]`` interpolate between embedding similarity and graph
        structure.
    :param tol: Convergence threshold for the Frobenius norm between
        consecutive score matrices.
    :param mask: Optional array broadcastable to ``(len(G1), len(G2))``.
        It is multiplied element-wise into both the initial scores and every
        iteration, so zeros exclude candidate node pairs.
    :param device: PyTorch device on which to run the similarity calculation,
        such as ``"cuda"``, ``"cuda:1"``, ``"mps"``, or ``"cpu"``.
    :return: A ``float32`` tensor of shape ``(len(G1), len(G2))`` containing
        the final node-pair similarity scores. The tensor remains on
        ``device``.
    :rtype: torch.Tensor
    :raises TypeError: If a graph or numeric argument has an invalid type.
    :raises ValueError: If a graph is empty, required attributes are missing
        or malformed, embedding dimensions do not match, or a numeric argument
        is outside its accepted range.
    :raises RuntimeError: If the requested PyTorch device is unavailable.

    .. note::
        Edge relation similarities are calculated from
        ``"relation_embeddings"`` but are not currently used in the update.
        Iteration progress is emitted to this module's logger at ``DEBUG``
        level. The returned tensor is created in inference mode and does not
        track gradients.
    """
    emb1, emb2, alpha, tol, mask, device = _validate_inputs(
        G1, G2, alpha, tol, mask, device
    )
    # S - current belief that node i corresponds to j
    S_0 = _distance_mtrx(emb1, emb2, 1, device=device)

    if mask is not None:
        S_0 *= mask

    adj_1 = torch.as_tensor(nx.to_numpy_array(G1), dtype=torch.float32, device=device)
    adj_2 = torch.as_tensor(nx.to_numpy_array(G2), dtype=torch.float32, device=device)

    p_1 = _transition_matrix(adj_1)
    p_2 = _transition_matrix(adj_2)

    S = S_0.clone()
    o = 0.0
    for iteration in range(100):
        S_struct = p_1.T @ S @ p_2

        max_val = torch.max(S_struct)
        if max_val.item() > 0:
            S_struct = S_struct / max_val

        if unbias_step:
            o = o + alpha * (1 - o)
            beta = alpha / o
            S_new = beta * S_struct + (1 - beta) * S_0
        else:
            S_new = alpha * S_struct + (1 - alpha) * S_0

        if mask is not None:
            S_new *= mask

        score_change = torch.linalg.vector_norm(S_new - S).item()
        logger.debug("IsoRank iteration %d: score change %.6g", iteration, score_change)
        # print(f"IsoRank iteration {iteration}: score change {score_change}")

        if score_change < tol:
            S = S_new
            break

        S = S_new
    return S
