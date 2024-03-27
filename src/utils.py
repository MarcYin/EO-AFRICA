import pkgutil
import importlib
import subprocess

pkg = {}
pkg_list = ['geojson','shapely','matplotlib',\
            'datetime','ee','mpl_toolkits','os','warnings','stats',\
            'geopandas','pyproj']
for p in pkg_list: pkg[p] = p
# special cases
pkg['arc']       = 'https://github.com/MarcYin/ARC/archive/refs/heads/main.zip'
pkg['jax']       = "jax[cpu]"

from pathlib import Path
from shapely import geometry
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress

from pyproj import Transformer

# code to pip load packages
# and then import. There are neater ways to make this 
# transportable
for p in pkg:
    if pkgutil.find_loader(p) is None:
        subprocess.call(f'(pip install -U {pkg[p]} > logs/{p}_install.log) &> logs/{p}_install_stderr.log',shell=True)
    globals()[p] = importlib.import_module(p, package=p)


warnings.filterwarnings('ignore') 



def biophysical_code(name: str):
    retval = {}
    retval['N']     =[0,100.,  "",             None,None];
    retval['cab']   =[1,100.,  "$\mu g/cm^2$", None,None];
    retval['cm']    =[2,10000.,"$g/cm^2$",     None,None];
    retval['cw']    =[3,10000.,"$cm$",         None,None];
    retval['lai']   =[4,100.,  "$m^2/m^2$",    0,7.];
    retval['ala']   =[5,100.,  "$^{\circ}$",   0,90.];
    retval['cbrown']=[6,1000., "",             0,1];
    if name in retval:
        return retval[name]
    return (None,None,None)

def plot_over_time(doys: np.array, post_bio_tensor: np.array, name: str,\
                  ALPHA=0.8, LINE_WIDTH=2, LAZY_EVALUATION_STEP=100):
    """Plot biophysical parameter over time for name (see biophysical_code())"""

    codes = biophysical_code(name);
    if codes[0] == None:
        return None
    
    plt.figure(figsize=(12, 6))
    plt.plot(
        doys,
        post_bio_tensor[
            ::LAZY_EVALUATION_STEP,
            codes[0],
        ].T
        / codes[1],
        "-",
        lw=LINE_WIDTH,
        alpha=ALPHA,
    )
    plt.ylabel(f"{str.upper(name)} {codes[2]}")
    plt.xlabel("Day of year")
    plt.show()

    
def plot_maps(doys: np.array, post_bio_tensor: np.array, mask: np.array, \
              name: str, SHRINK=0.8, symb=["w+","w1","w2","w3","w4"],Acoords=np.array([[72, 20],[ 8, 54]])):
    """Plot maps of biophysical parameters for name (see biophysical_code())"""

    codes = biophysical_code(name);
    if codes[0] == None:
        return None
    # min / max plotting
    if codes[3] == None:
        codes[3] = np.nanmin(post_bio_tensor[:, codes[0]].T / codes[1])
    if codes[4] == None:
        codes[4] = np.nanmax(post_bio_tensor[:, codes[0]].T / codes[1])
        
    # note that the biophysical variable is called LAI 
    # for historical reasons but may be any variable
    # its type is controlled by codes (name)
    lai = post_bio_tensor[:, codes[0]].T / codes[1]
    nrows = int(len(doys) / 5) + int(len(doys) % 5 > 0)
    fig, axs = plt.subplots(ncols=5, nrows=nrows, figsize=(20, 4 * nrows))
    axs = axs.ravel()

    for i in range(len(doys)):
        lai_map = np.zeros(mask.shape) * np.nan
        lai_map[~mask] = lai[i]
        im = axs[i].imshow(lai_map, vmin=codes[3], vmax=codes[4])
        fig.colorbar(im, ax=axs[i], shrink=SHRINK, label=f"{str.upper(name)} {codes[2]}")
        for j in range(int(Acoords.shape[0]/5)):
            axs[i].plot(SHRINK*Acoords.T[0,j*5:(j+1)*5],SHRINK*Acoords.T[1,j*5:(j+1)*5],symb[j])
        axs[i].set_title("DOY: %d" % doys[i])

    # remove empty plots
    for i in range(len(doys), len(axs)):
        axs[i].axis("off")
    plt.show()
    
def plot_area(post_bio_tensor,name,mask,buffer_n=2,geotransform=None,transformer=None,samplefile=None):
    
    #dataset=s2_lai,ylabel='LAI $(m^2/m^2)$',meas_name='LAI_measurement',quant='LAI'):
    n_code,scale,unit,ymin,ymax = biophysical_code(name)
 
    dataset_full = post_bio_tensor[:, n_code]
    temp = np.zeros(mask.shape + (dataset_full.shape[1], ))
    temp[~mask] = dataset_full
    dataset = temp

    #transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)


    ylabel = f'{name.upper()} {unit}'
    meas_name = f'{name.upper()}_measurement'
    quant = f'{name.upper()}'

    fig, axs = plt.subplots(nrows=1, ncols = 1, figsize=(20,10))
    ax = axs
    ax.set_title(f'Maximum predicted {quant}, showing sample locations',fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    mm = np.max(dataset,axis=2)/scale
    datamax = int(np.nanmax(mm))+1
    mm[mm==0] = np.nan

    all_data = []
    im = ax.imshow(mm,vmin=0,vmax=datamax)
    fig.colorbar(im, cax=cax, orientation='vertical')

    features = geojson.load(open(samplefile, 'r'))['features']

    # plot sample areas
    for i in range(1, 6):
        c =[]
        data = []
        for j in range(1, 6):
            for feature in features:
                field_id = feature['properties']['Name']
                if field_id == f'P{i}S{j}':
                    coord = feature['geometry']['coordinates'][:2]
                    geom = geometry.Point(coord)

                    Tcoord = transformer.transform(coord[0], coord[1])
                    Acoord = (Tcoord[0] - geotransform[0]) / geotransform[1], \
                            (Tcoord[1] - geotransform[3]) / geotransform[5]
                    Acoord = np.round(np.array(Acoord)).astype(int) 
                    c.append([Acoord[1], Acoord[0]])
                    # in case you have it inconsistent
                    try:
                        lai = feature['properties'][meas_name]
                    except:
                        meas_name = f'{name.capitalize()}_measurement'
                        lai = feature['properties'][meas_name]
                    data.append(lai)
        all_data.append(data)
        mask = np.max(dataset,axis=2)>0

        omask = mask.copy()

        c = np.array(c)
        for cc in c:
            for k in range(-buffer_n,buffer_n+1):
                for j in range(-buffer_n,buffer_n+1):
                    try:
                        mask[cc[0]+k,cc[1]+j] = 0
                    except:
                        pass
        mask = omask * ~mask
        ax.contour(mask)
        y,x = np.round(c.mean(axis=0)).astype(int)
        ax.text(x-4,y, f'sample {i}', color="red", fontsize=12)
        ax.plot(c.T[1],c.T[0],'w+')

    ax.set_axis_off()


def plot_biophys(post_bio_tensor,name,mask,t0s=None,t1s=None,t0a=None,t1a=None,\
                                           year=2000,doys=None,samplefile=None,buffer_n=None,\
                                           transformer=None,geotransform=None):
                 
    n_code,scale,unit,ymin,ymax = biophysical_code(name)
    
    dataset_full = post_bio_tensor[:, n_code]
    temp = np.zeros(mask.shape + (dataset_full.shape[1], ))
    temp[~mask] = dataset_full
    dataset = temp

    ylabel = f'{name.upper()} {unit}'
    meas_name = f'{name.upper()}_measurement'
    quant = f'{name.upper()}'

    features = geojson.load(open(samplefile, 'r'))['features']
    geoms = []
    
    datamax = np.nanmax(dataset/scale)
    # generate date fields for satellite observations
    s2_dates = [datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(i-1)) for i in doys]
    # interpolate to target dates
    sample_dates = np.array([int(d.strftime("%j")) for d in s2_dates])

    fig, axs = plt.subplots(nrows=5, ncols = 1, figsize=(16, 40))
    axs = axs.ravel()

    all_target_s2_pred = []
    all_data = []
    for i in range(1, 6):
        ax = axs[i-1]
        c =[]
        data = []
        for j in range(1, 6):
            for feature in features:
                field_id = feature['properties']['Name']
                if field_id == f'P{i}S{j}':
                    coord = feature['geometry']['coordinates'][:2]
                    geom = geometry.Point(coord)

                    Tcoord = transformer.transform(coord[0], coord[1])
                    Acoord = (Tcoord[0] - geotransform[0]) / geotransform[1], \
                        (Tcoord[1] - geotransform[3]) / geotransform[5]
                    Acoord = np.round(np.array(Acoord)).astype(int) 
                    c.append([Acoord[1], Acoord[0]])

                    try:
                        lai = feature['properties'][meas_name]
                    except:
                        meas_name = f'{name.capitalize()}_measurement'
                        lai = feature['properties'][meas_name]

                    data.append(lai)
                    dates = [datetime.datetime.strptime(i, '%Y%m%d') \
                             for i in feature['properties']['measurement_dates']]
                    target_dates = np.array([int(d.strftime("%j")) for d in dates])

        all_data.append(data)
        mask = np.max(dataset,axis=2)>0
        omask = mask.copy()

        c = np.array(c)
        for cc in c:
            for k in range(-buffer_n,buffer_n+1):
                for j in range(-buffer_n,buffer_n+1):
                    try:
                        mask[cc[0]+k,cc[1]+j] = 0
                    except:
                        pass
        mask = omask * ~mask
        # max measured for plot
        meas_max = np.round(np.max(np.max(data,axis=0)))

        _=ax.plot(dates, np.mean(data,axis=0).T,'r',lw=2)
        _=ax.plot(dates, np.mean(data,axis=0).T,'ro',label=f'mean observed {quant}')
        _=ax.plot(dates, np.mean(data,axis=0).T + np.std(data,axis=0).T,'r--',label=f'+/- 1 sd observed {quant}')
        _=ax.plot(dates, np.mean(data,axis=0).T - np.std(data,axis=0).T,'r--')

        _=ax.plot(dates, np.min(data,axis=0),'r-',lw=0.5,label=f'min/max observed {quant}')
        _=ax.plot(dates, np.max(data,axis=0),'r-',lw=0.5)
        ax.plot(s2_dates,np.mean(dataset[mask]/100.,axis=0),'k-',lw=2)
        ax.plot(s2_dates,np.mean(dataset[mask]/100.,axis=0),'ko',label=f'mean predicted {quant}')

        ax.plot(s2_dates,np.mean(dataset[mask]/100.,axis=0)+np.std(dataset[mask]/100.,axis=0),\
                'k--',label=f'+/- 1 sd predicted {quant}')
        ax.plot(s2_dates,np.mean(dataset[mask]/100.,axis=0)-np.std(dataset[mask]/100.,axis=0),\
                'k--')

        ax.plot(s2_dates,np.min(dataset[mask]/100.,axis=0),'k-',lw=0.5,label=f'min/max predicted {quant}')
        ax.plot(s2_dates,np.max(dataset[mask]/100.,axis=0),'k-',lw=0.5)
        ax.set_xlabel('date')
        ax.set_ylabel(ylabel)
        ax.set_title(f'location sample {i}',fontsize=12)
        ymax = np.max([datamax,meas_max])
        ax.set_ylim(0,ymax)

        # interpolate to target dates
        target_s2_pred = np.array([np.interp(target_dates,sample_dates,d) for d in (dataset[mask]/100.)]).T
        _=ax.plot(dates,np.mean(target_s2_pred,axis=1),'k+')
        all_target_s2_pred.append(target_s2_pred)

        ax.plot([t0s,t0s],[0,ymax],'r',label='start date')
        ax.plot([t1s,t1s],[0,ymax],'b',label='end date')

        ax.plot([t0a,t0a],[0,ymax],'r--',label='start of season')
        ax.plot([t1a,t1a],[0,ymax],'g--',label='end of growth')
        ax.legend()

    return datamax, all_target_s2_pred, all_data


def scatter_save(post_bio_tensor,name,mask,all_target_s2_pred, all_data,TAG,year):
                 
    n_code,scale,unit,ymin,ymax = biophysical_code(name)
    
    dataset_full = post_bio_tensor[:, n_code]
    temp = np.zeros(mask.shape + (dataset_full.shape[1], ))
    temp[~mask] = dataset_full
    dataset = temp

    ylabel = f'{name.upper()} {unit}'
    meas_name = f'{name.upper()}_measurement'
    quant = f'{name.upper()}'
    
    datamax = np.nanmax(dataset/scale)

    # plot limit
    m = datamax
    mean_pred = np.array([np.mean(d,axis=1) for d in all_target_s2_pred])
    mean_obs = np.mean(all_data,axis=0)
    std_pred = np.array([np.std(d,axis=1) for d in all_target_s2_pred])
    std_obs = np.std(all_data,axis=0)

    plt.figure(figsize=(6,6))
    plt.errorbar(mean_obs.ravel(),mean_pred.ravel(),std_obs.ravel(),std_pred.ravel(),'k.')
    plt.plot(mean_obs,mean_pred,'bo')

    plt.plot([0.,m],[0.,m],'r--')
    plt.xlabel(f'observed {name.upper()} ({unit})')
    plt.ylabel(f'S2-predicted {name.upper()} ({unit})')

    plt.xlim(0,m)
    plt.ylim(0,m)

    slope, intercept, r, p, se = linregress(mean_obs.ravel(),mean_pred.ravel())

    plt.title(f'{year} validation results\n\ny = ({intercept:.2} + {slope:.2} * x); '+\
              f' R = {r:.2};\n\nN = 45;   p = {p:.2};   se = {se:.2}\n')
    plt.plot([0.,m],[intercept,intercept+slope*m],'k--',\
             label=f'regression line')

    plt.plot(mean_obs[0],mean_pred[0],'bo',label=f'{name.upper()} ({unit})')

    _=plt.legend()

    np.savez(f'data/{TAG}_{name}.npz',mean_pred=mean_pred,mean_obs=mean_obs,std_pred=std_pred,std_obs=std_obs)


