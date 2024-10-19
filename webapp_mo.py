import marimo

__generated_with = "0.9.10"
app = marimo.App(width="full")


@app.cell
def __():
    import os
    import pickle
    import random

    import marimo as mo
    return mo, os, pickle, random


@app.cell
def __():
    # webapp_dir = "webapp/jp7xtqeu"
    # webapp_dir = "webapp/p9jmneyb"
    webapp_dir = "/local/scratch/stevens.994/sae-webapp/2dlebd60-original-analysis/original-generate/webapp"
    return (webapp_dir,)


@app.cell
def __(get_metadata, os, webapp_dir):
    neuron_indices = [
        int(name) for name in os.listdir(f"{webapp_dir}/neurons") if name.isdigit()
    ]
    neuron_indices = sorted(neuron_indices)

    metadatas = [get_metadata(i) for i in neuron_indices]
    return metadatas, neuron_indices


@app.cell
def __(mo):
    get_neuron_i, set_neuron_i = mo.state(0)
    get_neuron_i()
    return get_neuron_i, set_neuron_i


@app.cell
def __(mo, set_neuron_i):
    next_button = mo.ui.button(
        label="Next",
        on_change=lambda _: set_neuron_i(lambda v: v + 1),
    )

    prev_button = mo.ui.button(
        label="Previous",
        on_change=lambda _: set_neuron_i(lambda v: v - 1),
    )
    return next_button, prev_button


@app.cell
def __(mo, pickle, webapp_dir):
    def get_metadata(neuron: int):
        with open(f"{webapp_dir}/neurons/{neuron}/meta_data.pkl", "rb") as fd:
            return pickle.load(fd)

    def format_metadata(metadata: dict[str, float | int]):
        return mo.table([metadata])
    return format_metadata, get_metadata


@app.cell
def __(mo, next_button, prev_button):
    mo.hstack([prev_button, next_button])
    return


@app.cell
def __(get_neuron_i, mo, neuron_indices):
    mo.md(f"""Neuron {neuron_indices[get_neuron_i()]}""")
    return


@app.cell
def __(get_neuron_i, metadatas, mo):
    mo.ui.table([metadatas[get_neuron_i()]], selection=None)
    return


@app.cell
def __(get_neuron_i, mo, neuron_indices, webapp_dir):
    mo.image(
        f"{webapp_dir}/external/neurons/{neuron_indices[get_neuron_i()]}/highest_activating_images.png"
    )
    return


if __name__ == "__main__":
    app.run()
