


def add_line(data, color, ax=None, time=slice('2010', None), **kwargs):
    """select time period and plot mean tas"""

    print("Using `add_line` from `utils`.")
    
    if ax is None:
        ax = plt.gca()

    # get data    
    data = data.sel(time=time)

    co2 = data.co2
    tasmean = data.tas_anom.mean('ens')

    return ax.plot(co2, tasmean, color=color, **kwargs)



def add_year(ax, data, time, color, right=True):
    """add annotation for year"""    

    # convert time (2000) to a string ('2000')
    str_time = str(time)

    # get tas_anom and co2 at time
    tas_anom = data.tas_anom.sel(time=str(time)).mean('ens')
    co2 = data.co2.sel(time=str(time))

    x_offset = 3 if right else -3

    ha = 'left' if right else 'right'
    va = 'top' if right else 'bottom'


    ax.annotate(str_time,
                xy=(co2, tas_anom),
                xytext=(x_offset, -2), textcoords='offset points',
                va=va,
                ha=ha,
                fontsize=8,
                color=color)