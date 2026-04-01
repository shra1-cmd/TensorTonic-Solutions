def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    dot_prod = sum(a*b for a,b in zip(x1,x2))
    norm_x1 = math.sqrt(sum(a*a for a in x1))
    nowrm_x2 = math.sqrt(sum(a*a for a in x2))
    cosine_sim = dot_prod/(norm_x1*nowrm_x2)

    if label ==1:
        return (1 - cosine_sim)
    else:
        return float(max(0,cosine_sim - margin))