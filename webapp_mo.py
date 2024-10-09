import marimo

__generated_with = "0.9.4"
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
    webapp_dir = "web_app"
    s3_dir = "saeexplorer"
    return s3_dir, webapp_dir


@app.cell
def __(os, random, webapp_dir):
    neuron_indices = [
        int(name) for name in os.listdir(f"{webapp_dir}/neurons") if name.isdigit()
    ]
    random.shuffle(neuron_indices)
    return (neuron_indices,)


@app.cell
def __(mo):
    get_neuron_i, set_neuron_i = mo.state(0)
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
def __(get_metadata, get_neuron_i, mo, neuron_indices):
    mo.ui.table([get_metadata(neuron_indices[get_neuron_i()])], selection=None)
    return


@app.cell
def __(get_neuron_i, mo, neuron_indices, s3_dir):
    mo.image(
        f"{s3_dir}/neurons/{neuron_indices[get_neuron_i()]}/highest_activating_images.png"
    )
    return


if __name__ == "__main__":
    app.run()
