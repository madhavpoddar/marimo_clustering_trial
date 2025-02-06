import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full", layout_file="layouts/clustering.slides.json")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import numpy as np
    import os
    from drawdata import ScatterWidget
    import torch
    from torchvision import datasets, transforms
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import base64
    from io import BytesIO
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    return (
        AnnotationBbox,
        BytesIO,
        Image,
        OffsetImage,
        PCA,
        ScatterWidget,
        TSNE,
        alt,
        base64,
        datasets,
        mo,
        np,
        os,
        pd,
        plt,
        torch,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to Clustering

        ---

        **_Clustering is a method of organizing data into meaningful groups based on similarity._**

        ---

        But instead of data, let us begin with a task of wardrobe management.
        """
    )
    return


@app.cell
def _(mo, pd):
    df_difficulty_opts = pd.DataFrame({
        "str_option": ["Easy", "Medium", "Hard"],
        "N_images": [30, 100, 250],
        "tsne_perplexity": [2, 1.0, 1.4],
        "img_size": [8, 2, 2]
    })
    df_difficulty_opts["str_radio"] = df_difficulty_opts.apply(lambda row: row["str_option"]+" ("+str(row["N_images"])+" items)", axis=1)


    N_images_options_radio = mo.ui.radio(options=list(df_difficulty_opts["str_radio"]), value=list(df_difficulty_opts["str_radio"])[0])
    return N_images_options_radio, df_difficulty_opts


@app.cell(hide_code=True)
def _(N_images_options_radio, mo, os):
    mo.vstack(
    [
        mo.md("Imagine you work at a company that designs wardrobes. A client shares their online clothing purchase records, and you have a tool to gather one image for each clothing item. Your task is to group these clothes based on their similarity to determine the right number and size of cabinets."),
        mo.image(os.path.join("img", "Fashion-MNIST-dataset-Activeloop-Platform-visualization-image.png")),
        mo.md("Select your difficulty level for this task:"),
        N_images_options_radio
    ]
    )
    return


@app.cell(hide_code=True)
def _(
    N_images_options_radio,
    TSNE,
    datasets,
    df_difficulty_opts,
    mo,
    np,
    plt,
    transforms,
):
    # Load FashionMNIST dataset from PyTorch
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    # Extract the first N_images and labels
    images = []
    labels = []
    img_size = 8

    selected_option_index = df_difficulty_opts.index[df_difficulty_opts['str_radio'] == N_images_options_radio.value][0]
    N_images = df_difficulty_opts.iloc[selected_option_index]["N_images"]
    tsne_perplexity = df_difficulty_opts.iloc[selected_option_index]["tsne_perplexity"]

    for i in range(N_images):
        image, label = dataset[i]

        # Flatten for t-SNE
        images.append(image.view(-1).numpy())  
        labels.append(label)

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
    tsne_result = tsne.fit_transform(np.array(images))

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    # scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', alpha=0.6)

    # Scale the plot to fit images correctly
    x_min, x_max = tsne_result[:, 0].min() - img_size, tsne_result[:, 0].max() + img_size
    y_min, y_max = tsne_result[:, 1].min() - img_size, tsne_result[:, 1].max() + img_size
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Plot images at their t-SNE coordinates
    for i in range(N_images):
        img = dataset[i][0].squeeze().numpy()  # Convert tensor image to numpy array
        x, y = tsne_result[i, 0], tsne_result[i, 1]

        # Display images at scatter plot coordinates with proper scaling
        ax.imshow(img, cmap='gray', extent=(x-img_size, x+img_size, y-img_size, y+img_size), aspect='auto')

    # Remove axes for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # plt.show()
    mo.vstack([mo.mpl.interactive(plt.gcf())])
    return (
        N_images,
        ax,
        dataset,
        fig,
        i,
        image,
        images,
        img,
        img_size,
        label,
        labels,
        selected_option_index,
        transform,
        tsne,
        tsne_perplexity,
        tsne_result,
        x,
        x_max,
        x_min,
        y,
        y_max,
        y_min,
    )


if __name__ == "__main__":
    app.run()
