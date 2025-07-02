def topoplot(
    data,
    montage="standard_1020",
    vmin=None,
    vmax=None,
    mask=None,
    title=None,
    cmap=None,
    n_colors=100,
    cbar_title=None,
    cbar_ticks=None,
    figsize=(4, 4),
    dpi=80,
    fontsize=14,
    **kwargs,
):
    """
    Topoplot.

    This is a wrapper around :py:func:`mne.viz.plot_topomap`.

    For more details, please refer to this `example notebook
    <https://github.com/raphaelvallat/yasa/blob/master/notebooks/15_topoplot.ipynb>`_.

    .. versionadded:: 0.4.1

    Parameters
    ----------
    data : :py:class:`pandas.Series`
        A pandas Series with the values to plot. The index MUST be the channel
        names (e.g. ['C4', 'F4'] or ['C4-M1', 'C3-M2']).
    montage : str
        The name of the montage to use. Valid montages can be found at
        :py:func:`mne.channels.make_standard_montage`.
    vmin, vmax : float
        The minimum and maximum values of the colormap. If None, these will be
        defined based on the min / max values of ``data``.
    mask : :py:class:`pandas.Series`
        A pandas Series indicating the significant electrodes. The index MUST
        be the channel names (e.g. ['C4', 'F4'] or ['C4-M1', 'C3-M2']).
    title : str
        The plot title.
    cmap : str
        A matplotlib color palette. A list of color palette can be found at:
        https://seaborn.pydata.org/tutorial/color_palettes.html
    n_colors : int
        The number of colors to discretize the color palette.
    cbar_title : str
        The title of the colorbar.
    cbar_ticks : list
        The ticks of the colorbar.
    figsize : tuple
       Width, height in inches.
    dpi : int
        The resolution of the plot.
    fontsize : int
        Global font size of all the elements of the plot.
    **kwargs : dict
        Other arguments that are passed to :py:func:`mne.viz.plot_topomap`.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        Matplotlib Figure

    Examples
    --------

    1. Plot all-positive values

    .. plot::

        >>> import yasa
        >>> import pandas as pd
        >>> data = pd.Series([4, 8, 7, 1, 2, 3, 5],
        ...                  index=['F4', 'F3', 'C4', 'C3', 'P3', 'P4', 'Oz'],
        ...                  name='Values')
        >>> fig = yasa.topoplot(data, title='My first topoplot')

    2. Plot correlation coefficients (values ranging from -1 to 1)

    .. plot::

        >>> import yasa
        >>> import pandas as pd
        >>> data = pd.Series([-0.5, -0.7, -0.3, 0.1, 0.15, 0.3, 0.55],
        ...                  index=['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'Pz'])
        >>> fig = yasa.topoplot(data, vmin=-1, vmax=1, n_colors=8,
        ...                     cbar_title="Pearson correlation")
    """
    # Increase font size while preserving original
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams.update({"savefig.bbox": "tight"})
    plt.rcParams.update({"savefig.transparent": "True"})

    # Make sure we don't do any in-place modification
    assert isinstance(data, pd.Series), "`data` must be a Pandas Series"
    data = data.copy()

    # Add mask, if present
    if mask is not None:
        assert isinstance(mask, pd.Series), "`mask` must be a Pandas Series"
        assert mask.dtype.kind in "bi", "`mask` must be True/False or 0/1."
    else:
        mask = pd.Series(1, index=data.index, name="mask")

    # Convert to a dataframe (col1 = values, col2 = mask)
    data = data.to_frame().join(mask, how="left")

    # Preprocess channel names: C4-M1 --> C4
    data.index = data.index.str.split("-").str.get(0)

    # Define electrodes coordinates
    Info = mne.create_info(data.index.tolist(), sfreq=100, ch_types="eeg")
    Info.set_montage(montage, match_case=False, on_missing="ignore")
    chan = Info.ch_names

    # Define vmin and vmax
    if vmin is None:
        vmin = data.iloc[:, 0].min()
    if vmax is None:
        vmax = data.iloc[:, 0].max()

    # Choose and discretize colormap
    if cmap is None:
        if vmin < 0 and vmax <= 0:
            cmap = "mako"
        elif vmin < 0 and vmax > 0:
            cmap = "Spectral_r"
        elif vmin >= 0 and vmax > 0:
            cmap = "rocket_r"

    cmap = ListedColormap(sns.color_palette(cmap, n_colors).as_hex())

    if "sensors" not in kwargs:
        kwargs["sensors"] = False
    if "res" not in kwargs:
        kwargs["res"] = 256
    if "names" not in kwargs:
        kwargs["names"] = chan
    if "mask_params" not in kwargs:
        kwargs["mask_params"] = dict(marker=None)

    # Hidden feature: if names='values', show the actual values.
    if kwargs["names"] == "values":
        kwargs["names"] = data.iloc[:, 0][chan].round(2).to_numpy()

    # Start the plot
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if "show_names" in kwargs:
            kwargs.pop("show_names")
        im, _ = mne.viz.plot_topomap(
            data=data.iloc[:, 0][chan],
            pos=Info,
            vlim=(vmin, vmax),
            mask=data.iloc[:, 1][chan],
            cmap=cmap,
            show=False,
            axes=ax,
            **kwargs,
        )

        if title is not None:
            ax.set_title(title)

        # Add colorbar
        if cbar_title is None:
            cbar_title = data.iloc[:, 0].name

        cax = fig.add_axes([0.95, 0.3, 0.02, 0.5])
        cbar = fig.colorbar(im, cax=cax, ticks=cbar_ticks, fraction=0.5)
        cbar.set_label(cbar_title)

        # Revert font-size
        plt.rcParams.update({"font.size": old_fontsize})
    return fig