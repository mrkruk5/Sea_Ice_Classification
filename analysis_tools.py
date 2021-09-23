import os
from osgeo import osr, gdal
import numpy as np
from matplotlib import pylab
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import product_info as pinfo
from calibration import calibrate, denoise


def plot_history(hist: dict, path_save):
    fig = plt.figure()
    plt.plot(hist['loss'], 'bo-')
    plt.plot(hist['val_loss'], 'rx--')
    # plt.title('Loss vs. epochs')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.xticks(np.arange(0, len(hist['loss']), step=2))
    plt.legend(['Training Loss', 'Testing Loss'])
    plt.grid()
    fig.savefig(os.path.join(path_save, 'loss_history.png'))
    plt.pause(0.5)
    plt.close(fig)


def get_sar_scene_data(img_files, pre_processing, fill):
    img = []
    tx = None
    xticks, xlabels, yticks, ylabels = [], [], [], []
    for file in img_files:
        path_root = '/'.join(file.split('/')[:-1])
        path_pif = os.path.join(path_root, 'product.xml')
        path_sigma = os.path.join(path_root, 'lutSigma.xml')
        src_ds = gdal.Open(file)
        src_projection = src_ds.GetProjection()
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(src_projection)
        sar_img = src_ds.GetRasterBand(1).ReadAsArray()
        rows, cols = sar_img.shape
        sar_extents = pinfo.get_extents(path_pif, rows, cols)
        geo_min = np.min(sar_extents, axis=0)
        geo_max = np.max(sar_extents, axis=0)
        noise = pinfo.get_noise(path_pif)
        sigma = pinfo.get_look_up_table(path_sigma)

        # Warp source image to get geo-transform of warped result.
        options = gdal.WarpOptions(
            format='VRT',
            outputBounds=(*geo_min, *geo_max),
            width=cols,
            height=rows,
            dstSRS='EPSG:4326'
        )
        if pre_processing == 'raw':
            warp_ds = gdal.Warp(
                '',
                src_ds,
                options=options
            )
        else:
            if 'cal' in pre_processing:
                sar_img = calibrate(sar_img, sigma)
            if 'denoise' in pre_processing:
                sar_img = denoise(sar_img, noise, fill)
            tmp_file = os.path.join(path_root, 'tmp.tif')
            driver = gdal.GetDriverByName('GTiff')
            proc_ds = driver.Create(tmp_file, cols, rows, 1,
                                    gdal.GDT_Float32)
            proc_ds.SetGCPs(src_ds.GetGCPs(), src_srs.ExportToWkt())
            proc_ds.SetProjection(src_srs.ExportToWkt())
            proc_ds.GetRasterBand(1).WriteArray(sar_img)
            proc_ds.FlushCache()
            warp_ds = gdal.Warp(
                '',
                proc_ds,
                options=options
            )
            os.remove(tmp_file)

        tmp = warp_ds.GetRasterBand(1).ReadAsArray()
        img.append(np.expand_dims(tmp, axis=0))

        if len(xticks) == 0:
            tx = gdal.Transformer(src_ds, warp_ds, ['METHOD=GCP_TPS'])
            for gcp in src_ds.GetGCPs():
                xticks.append(gcp.GCPPixel)
                xlabels.append(gcp.GCPX)
                yticks.append(gcp.GCPLine)
                ylabels.append(gcp.GCPY)
    img = np.concatenate(img, axis=0)

    pixel_coords = np.column_stack((xticks, yticks))
    proj_pixel_coords, success = tx.TransformPoints(0, pixel_coords)
    proj_xticks = np.array([coord[0] for coord in proj_pixel_coords])
    proj_yticks = np.array([coord[1] for coord in proj_pixel_coords])

    xlabels_sort_idx = np.argsort(xlabels)
    ylabels_sort_idx = np.argsort(ylabels)

    proj_xticks_sorted = proj_xticks[xlabels_sort_idx]
    proj_yticks_sorted = proj_yticks[ylabels_sort_idx]
    xlabels_sorted = np.round(xlabels, 2)[xlabels_sort_idx]
    ylabels_sorted = np.round(ylabels, 2)[ylabels_sort_idx]
    return img, proj_xticks_sorted, xlabels_sorted, \
           proj_yticks_sorted, ylabels_sorted


def plot_img(label_list, pred_list, title_list, img_files,
             sar_scenes, indices, mode, pre_processing, path):
    params = {
        'axes.titlesize': 'xx-large',
        'axes.labelsize': 'xx-large',
        'xtick.labelsize': 'xx-large',
        'ytick.labelsize': 'xx-large'
    }
    pylab.rcParams.update(params)
    SAR_FILL_VALUE = 0

    def color_bar(mappable, ticks=None):
        ax = mappable.axes
        map_fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        return map_fig.colorbar(mappable, cax=cax, ticks=ticks)

    cmap_ice = plt.get_cmap('rainbow')
    bounds = np.append(np.unique(label_list), label_list.max() + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap_ice.N)
    num_plots = 0
    for ind_list, img_name in zip(indices, list(sar_scenes)):
        img_labels = label_list[np.array(ind_list)]
        img_preds = pred_list[np.array(ind_list)]
        img_titles = title_list[np.array(ind_list)]
        lat, lon = [], []
        for title in img_titles:
            title_tokens = title.split('_')
            lat.append(float(title_tokens[-2]))
            lon.append(float(title_tokens[-1]))

        search_str = '_'.join(img_name.split('_')[2: 4])
        idx = np.where(np.char.find(img_files, search_str) >= 0)
        assert np.unique(idx[0]).size
        img, xticks, xlabels, yticks, ylabels = get_sar_scene_data(
            img_files[idx], pre_processing, SAR_FILL_VALUE
        )
        tick_idx = np.round(np.linspace(0, len(xticks) - 1, 5)).astype(int)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5), dpi=100)
        masked_img = np.ma.masked_where(img == SAR_FILL_VALUE, img)
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='black')
        sar_img = ax1.imshow(masked_img[0], cmap=cmap)
        ax1.set_title(img_name)
        ax1.set_xticks(xticks[tick_idx])
        ax1.set_xticklabels(xlabels[tick_idx])
        ax1.set_xlabel('Lon')
        ax1.set_yticks(yticks[tick_idx])
        ax1.set_yticklabels(ylabels[tick_idx])
        ax1.set_ylabel('Lat')
        color_bar(sar_img)

        ax2.scatter(lon, lat, s=4, c=img_labels, cmap=cmap_ice, norm=norm)
        ax2.set_title('CIS Labels')
        ax2.set_xlim(left=np.min(xlabels[tick_idx]),
                     right=np.max(xlabels[tick_idx]))
        ax2.set_xlabel('Lon')
        ax2.set_ylim(bottom=np.min(ylabels[tick_idx]),
                     top=np.max(ylabels[tick_idx]))
        ax2.set_ylabel('Lat')

        scatter3 = ax3.scatter(lon, lat, s=4,
                               c=img_preds, cmap=cmap_ice, norm=norm)
        ax3.set_title('Predicted Labels')
        ax3.set_xlim(left=np.min(xlabels[tick_idx]),
                     right=np.max(xlabels[tick_idx]))
        ax3.set_xlabel('Lon')
        ax3.set_ylim(bottom=np.min(ylabels[tick_idx]),
                     top=np.max(ylabels[tick_idx]))
        ax3.set_ylabel('Lat')
        color_bar(scatter3, ticks=np.unique(label_list))

        fig.tight_layout(pad=0.2)
        fig.savefig(
            os.path.join(path, img_name + '_' + mode + '_scatter_plot.png'),
            bbox_inches='tight'
        )
        plt.pause(0.5)
        plt.close(fig)
        num_plots += 1

        if num_plots == 3:
            break
    return


def unique_titles(titles):
    unique = []
    for t in titles:
        unique.append('_'.join(t.split('_')[:-2]))
    unique = sorted(set(unique))
    return unique


def crops_to_scene(titles, img_name_set):
    idx = []
    for i in range(len(img_name_set)):
        idx.append([])
    for i, t in enumerate(titles):
        for j, search_str in enumerate(img_name_set):
            if search_str in t:
                idx[j].append(i)
    return idx


def print_label_acc(label_types, labels, right_wrong):
    for label in label_types:
        label_idx = np.where(labels == label)
        label_rw = right_wrong[label_idx]
        label_acc = np.sum(label_rw) / label_rw.size
        print(f'Label {label}: Accuracy = {label_acc:.4}')


def print_monthly_acc(months, titles, right_wrong, mode):
    for month in months:
        month_idx = np.where(np.char.find(titles, month) >= 0)[0]
        month_rw = right_wrong[month_idx]
        month_acc = np.sum(month_rw) / month_rw.size
        print(f'Month {month} makes '
              f'{(month_idx.size / titles.size) * 100:>5.2f}% '
              f'of {mode} samples. '
              f'Accuracy = {month_acc * 100:<5.2f}')


def print_label_num_per_month(label_types, labels, months, titles):
    n = labels.size
    n_digits = int(np.log10(n)) + 1
    print(f'Month & {label_types}')
    for month in months:
        month_idx = np.char.find(titles, month) >= 0
        print(month, end=' ')
        for label in label_types:
            label_idx = labels == label
            idx = np.logical_and(month_idx, label_idx)
            print(f'{np.sum(idx):>{n_digits}} ', end='')
        print()


def print_label_acc_per_month(label_types, labels, right_wrong, months, titles):
    print(f'Month & {label_types}')
    for month in months:
        month_idx = np.char.find(titles, month) >= 0
        print(month, end=' ')
        for label in label_types:
            label_idx = labels == label
            idx = np.where(np.logical_and(month_idx, label_idx))[0]
            if idx.size == 0:
                print('  -  ', end=' ')
            else:
                # Right/wrong in a month for each label
                ml_rw = right_wrong[idx]
                ml_acc = np.sum(ml_rw) / ml_rw.size
                print(f'{ml_acc * 100:>5.2f} ', end='')
        print()
