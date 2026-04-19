import os
import h5py
import time
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import pygrib as pg
from datetime import datetime,timedelta
from scipy.interpolate import griddata
from multiprocessing import Pool, cpu_count

__all__ = ["make_input", "print_dataarray"]

parser = argparse.ArgumentParser()
parser.add_argument('--time', type=str, default='0', help="Start time")
#parser.add_argument('--outdir', type=str, default="data", help="Output dir for input.nc")
#parser.add_argument('--patch', type=int, default=2, help="Patch for OLR interpolating")
#parser.add_argument('--pad', type=int, default=10, help="Padpix for OLR interpolating")
args = parser.parse_args()

unit_scale = dict(gh=9.80665, tp=1000)
levels_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

cra_names = dict(
    z=dict(prefix='GPH', short_name='gh', long_name="geopotential", levels=levels_13),
    t=dict(prefix='TEM', short_name='t', long_name="temperature", levels=levels_13),
    u=dict(prefix='WIU', short_name='u', long_name="u_component_of_wind", levels=levels_13),
    v=dict(prefix='WIV', short_name='v', long_name="v_component_of_wind", levels=levels_13),
    q=dict(prefix='SHU', short_name='q', long_name="specific_humidity", levels=levels_13),
    t2m=dict(prefix='SURFACE', short_name='2t', long_name="2m_temperature", levels=[1]),
    d2m=dict(prefix='SINGLEA', short_name='2d', long_name="2m_dewpoint_temperature", levels=[1]),
    sst=dict(prefix='SST', short_name='sst', long_name="sea_surface_temperature", levels=[1]),
    ttr=dict(prefix='OLR', short_name='olr', long_name="top_net_thermal_radiation", levels=[1]),
    u10m=dict(prefix='SINGLEA', short_name='10u', long_name="10m_u_component_of_wind", levels=[1]),
    v10m=dict(prefix='SINGLEA', short_name='10v', long_name="10m_v_component_of_wind", levels=[1]),
    u100m=dict(prefix='SINGLE', short_name='100u', long_name="100m_u_component_of_wind", levels=[1]),
    v100m=dict(prefix='SINGLE', short_name='100v', long_name="100m_v_component_of_wind", levels=[1]),    
    msl=dict(prefix='SINGLE', short_name='prmsl', long_name="mean_sea_level_pressure", levels=[1]),    
    tcwv=dict(prefix='SINGLE', short_name='pwat', long_name="total_column_water_vapour", levels=[1]),    
    tp=dict(prefix='SINGLE', short_name='pwat', long_name="total_precipitation", levels=[1]),    
)
CRA40_end = 'GLB_1P00_DAY'

def level_to_channel(ds, short_name, l0=1000):
    if len(ds.level) == 1:
        channel = [short_name]
    else:
        if ds.level.data[0] != l0:
            ds = ds.reindex(level=ds.level[::-1])
        channel = [f'{short_name}{lvl}' for lvl in ds.level.data]
    ds.attrs = {}  
    ds.name = "data"   
    ds = ds.rename({'level': 'channel'})
    ds = ds.assign_coords(channel=channel)  
    return ds

def print_dataarray(ds, msg='', n=10):
    tid = np.arange(0, ds.shape[0])
    tid = np.append(tid[:n], tid[-n:])    
    v = ds.isel(time=tid)
    msg += f"short_name: {ds.name}, shape: {ds.shape}, value: {v.values.min():.3f} ~ {v.values.max():.3f}"
    
    if 'lat' in ds.dims:
        lat = ds.lat.values
        msg += f", lat: {lat[0]:.3f} ~ {lat[-1]:.3f}"
    if 'lon' in ds.dims:
        lon = ds.lon.values
        msg += f", lon: {lon[0]:.3f} ~ {lon[-1]:.3f}"   

    if "level" in v.dims and len(v.level) > 1:
        for lvl in v.level.values:
            x = v.sel(level=lvl).values
            msg += f"\nlevel: {lvl:04d}, value: {x.min():.3f} ~ {x.max():.3f}"

    if "channel" in v.dims and len(v.channel) > 1:
        for ch in v.channel.values:
            x = v.sel(channel=ch).values
            msg += f"\nchannel: {ch}, value: {x.min():.3f} ~ {x.max():.3f}"

    print(msg)

def load_cra(file_name, short_name, new_name, levels=[]):

    try:
        ds = pg.open(file_name)

        if len(levels) > 1:
            data = ds.select(shortName=short_name, level=levels)
        else:
            data = ds.select(shortName=short_name)
    except:
        print(f"Load {short_name} failed !")
        return 
    
    for v in data:
        lats = v.distinctLatitudes
        lons = v.distinctLongitudes
        time_str = f'{v.dataDate}'

    # msg = f"time_str: {time_str}"
    # msg += f"lat: {len(lats)}, {lats[0]} ~ {lats[-1]}"
    # msg += f"lon: {len(lons)}, {lons[0]} ~ {lons[-1]}"
    # print(msg)

    imgs = np.zeros((len(levels), len(lats), len(lons)), dtype=np.float32)
    # assert len(data) == len(levels), (len(data))
    
    for v in data:  
        img, _, _ = v.data() 
        level = int(v.level)
        
        if len(levels) == 1:
            imgs[0] = img
        else:
            i = levels.index(level)
            imgs[i] = img

    init_times = [pd.to_datetime(time_str)]
    data = imgs[None] * unit_scale.get(short_name, 1)
    
    v = xr.DataArray(
        name=new_name,
        data=data,
        dims=['time', 'level', 'lat', 'lon'],
        coords={
            'time': init_times, 
            'level': levels, 
            'lat': lats, 
            'lon': lons
        },
    )
    return v

def load_cra40land(file_name, short_name, new_name, levels=[]):

    try:
        data = pg.open(file_name).select()
    except:
        print(f"Load {short_name} failed !")
        return 
    
    # assert len(data) == len(levels), (len(data))
    for v in data:  
        img, lats, lons = v.data() 
        lats = lats[:, 0]
        lons = lons[0, :]
        imgs = np.zeros((len(levels), len(lats), len(lons)), dtype=np.float32)
        imgs[0] = img
        # from IPython import embed; embed()
        time_str = str(v.dataDate)
        init_times = [pd.to_datetime(time_str)]
        new_data = imgs[None] * unit_scale.get(short_name, 1)
        # msg = f"{time_str}, img: {new_data.shape}, {new_data.min():.3f} ~  {new_data.max():.3f}, "
        # msg += f"lat: {len(lats)}, {lats[0]} ~ {lats[-1]}, "
        # msg += f"lon: {len(lons)}, {lons[0]} ~ {lons[-1]}"
        # print(msg)

        v = xr.DataArray(
            name=new_name,
            data=new_data,
            dims=['time', 'level', 'lat', 'lon'],
            coords={
                'time': init_times, 
                'level': levels, 
                'lat': lats, 
                'lon': lons
            },
        )
        return v

def load_sst(file_name, short_name, new_name, levels=[]):
    v = xr.open_dataarray(file_name)
    v.name = new_name
    v = v.expand_dims({'level': levels}, axis=0) 
    v = v.assign_coords(time=pd.to_datetime(file_name.split('.nc')[0][-8:])).expand_dims('time')
    return v

def load_ttr(file_path, short_name, new_name, levels=[]):
    data_name = ['OLR_A','OLR_D']
    file = h5py.File(file_path, 'r')
    data_out = []
    patch=2
    pad=10
    for step in range(2):
        data = file[data_name[step]][:]
        patch_size = [data.shape[0]//patch,data.shape[1]//patch]
        arr_out = np.zeros([data.shape[0],data.shape[1]])
        if patch == 1:  #不分区，慢
            points = np.argwhere(data != 32767)
            grid_x, grid_y = np.mgrid[0:data.shape[0], 0:data.shape[1]]
            arr_out = griddata(points, data[points[:, 0], points[:, 1]], (grid_x, grid_y), method='linear')#插值
        else:           #并行分区插值，快3倍
            input_data = []
            for i in range(patch):
                for j in range(patch):
                    coord = [[patch_size[0]*i,patch_size[0]*(i+1)],[patch_size[1]*j,patch_size[1]*(j+1)]]
                    cdout = [[patch_size[0]*i,patch_size[0]*(i+1)],[patch_size[1]*j,patch_size[1]*(j+1)]]
                    for k in range(2):
                        coord[k][0] = coord[k][0]-pad if coord[k][0]>0 else coord[k][0]
                        coord[k][1] = coord[k][1]+pad if coord[k][1]<data.shape[k] else coord[k][1]
                        cdout[k][0] = pad if cdout[k][0]>0 else 0
                        cdout[k][1] = data.shape[k]//patch if cdout[k][1]<data.shape[k] else (data.shape[k]//patch)+pad
                    arr = data[coord[0][0]:coord[0][1],coord[1][0]:coord[1][1]]
                    arr = np.pad(arr, pad_width=1, mode='constant', constant_values=data.min())
                    points = np.argwhere(arr != 32767)
                    grid_x, grid_y = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
                    input_data.append((points, arr[points[:,0],points[:,1]], (grid_x, grid_y), 'linear'))
            pool = Pool(processes=4)
            results = pool.starmap(griddata, input_data)
            pool.close()
            pool.join()
            for i in range(patch):#分区插值
                for j in range(patch):
                    coord = [[patch_size[0]*i,patch_size[0]*(i+1)],[patch_size[1]*j,patch_size[1]*(j+1)]]
                    cdout = [[patch_size[0]*i,patch_size[0]*(i+1)],[patch_size[1]*j,patch_size[1]*(j+1)]]
                    for k in range(2):
                        coord[k][0] = coord[k][0]-pad if coord[k][0]>0 else coord[k][0]
                        coord[k][1] = coord[k][1]+pad if coord[k][1]<data.shape[k] else coord[k][1]
                        cdout[k][0] = pad if cdout[k][0]>0 else 0
                        cdout[k][1] = data.shape[k]//patch if cdout[k][1]<data.shape[k] else (data.shape[k]//patch)+pad
                    arr = data[coord[0][0]:coord[0][1],coord[1][0]:coord[1][1]]
                    arr = np.pad(arr, pad_width=1, mode='constant', constant_values=data.min())
                    arr_new = results[i*patch+j]
                    arr_new = arr_new[1:(arr.shape[0]-1),1:(arr.shape[1]-1)]
                    arr_out[patch_size[0]*i:patch_size[0]*(i+1),patch_size[1]*j:patch_size[1]*(j+1)] = arr_new[cdout[0][0]:cdout[0][1],cdout[1][0]:cdout[1][1]]
        data_out.append(arr_out)
    data_out = np.mean(np.array(data_out),0)#日夜平均
    data = np.zeros([data_out.shape[0],data_out.shape[1]])
    data[:,:3600] = data_out[:,3600:]#经度对齐
    data[:,3600:] = data_out[:,:3600]
    new_shape = (180, 360)#均值下采样
    data_out = np.mean(data.reshape(new_shape[0], data.shape[0] // new_shape[0], new_shape[1], data.shape[1] // new_shape[1]), axis=(1, 3)).reshape((1,180,360))
    #data_out = np.zeros([1,180,360])#test
    for i in range(file_path.count('_')):
        period = file_path.split('_')[i]
        if len(period) == 8 and period.isdigit():
            break
    coords = {
        'lat': [np.float32(x) for x in [89.5-i for i in range(180)]],
        'lon': [np.float32(x) for x in [0.5+i for i in range(360)]],
        'time': pd.date_range(period, periods=1, freq='D')
    }
    v = xr.DataArray(data_out, coords=coords, dims=['time','lat','lon'])
    v.name = new_name
    v = v.expand_dims({'level': levels}, axis=0)
    return -v

def get_file_name(data_dir, prefix):
    for file_name in os.listdir(data_dir):
        if 'CRA40' in file_name:
            if CRA40_end in file_name and f"_{prefix}_" in file_name:
                return os.path.join(data_dir, file_name)
        else:
            if f"_{prefix}_" in file_name:
                return os.path.join(data_dir, file_name)
            elif f"-{prefix}-" in file_name:
                return os.path.join(data_dir, file_name)
    return ""

def data_check(data_dir):
    for dirs in data_dir:
        if not os.path.exists(dirs):
            print('waiting for rawdatadir:',dirs,datetime.now())
            return True
    for new_name, cra_name in cra_names.items():
        prefix = cra_name['prefix']
        if new_name == "t2m":
            file_name = get_file_name(data_dir[0], prefix)
        elif new_name == "sst":
            file_name = get_file_name(data_dir[2], prefix)
        elif new_name == "ttr":
            file_name = get_file_name(data_dir[1], prefix)
        else:
            file_name = get_file_name(data_dir[0], prefix)
        if not os.path.exists(file_name):
            print('waiting for rawdata',prefix,datetime.now())
            return True
    print('rawdata ready',datetime.now())
    return False

def make_single(data_dir, degree=1.5):
    ds = []
    lat = np.arange(90, -90-degree , -degree)
    lon = np.arange(0, 360, degree)

    while data_check(data_dir):
        time.sleep(300)

    for new_name, cra_name in cra_names.items():
        prefix = cra_name['prefix']
        short_name = cra_name['short_name']
        levels = cra_name['levels']

        if new_name == "t2m":
            file_name = get_file_name(data_dir[0], prefix)
            v = load_cra40land(file_name, short_name, new_name, levels)
        elif new_name == "sst":
            file_name = get_file_name(data_dir[2], prefix)
            v = load_sst(file_name, short_name, new_name, levels)
        elif new_name == "ttr":
            file_name = get_file_name(data_dir[1], prefix)
            v = load_ttr(file_name, short_name, new_name, levels)
        else:
            file_name = get_file_name(data_dir[0], prefix)
            v = load_cra(file_name, short_name, new_name, levels)

        v = level_to_channel(v, new_name, l0=1000)
        if v.lat.data[0] < 0:
            v = v.reindex(lat=v.lat[::-1])        
        v = v.interp(lat=lat, lon=lon, kwargs={"fill_value": "extrapolate"})

        v.data = v.data.astype(np.float32)

        # zero tp
        if new_name == "tp":
            v = v * 0

        ds.append(v)

    ds = xr.concat(ds, 'channel')
    #print_dataarray(ds)
    return ds

def make_input(input_time):
    time_need = []
    time_now = datetime.now()-timedelta(days=1) if input_time == '0' else datetime.strptime(input_time, '%Y%m%d')
    time_need.append(time_now.strftime('%Y%m%d'))
    time_need.append((time_now-timedelta(days=1)).strftime('%Y%m%d'))
    print('Make x from',time_need)

    data_dir = ["/*/CRA40_RELEASE","/*/FY3E","/*/SST"]
    file_names = [["" for _ in range(2)] for _ in range(3)]
    for i in range(len(data_dir)):
        for time_str in time_need:
            file_names[i].append(os.path.join(data_dir[i], time_str[:4],time_str))
        file_names[i] = sorted(file_names[i])[-2:]
    #print(f"Make x1 from {[row[0] for row in file_names]}")
    x1 = make_single([row[0] for row in file_names])
    #print(f"Make x2 from {[row[1] for row in file_names]}")
    x2 = make_single([row[1] for row in file_names])
    input = xr.concat([x1, x2], "time")
    input.to_netcdf('data/input.nc')
    print('Data ready!',datetime.now())
    return input

if __name__ == "__main__":
    input = make_input(args.time)
