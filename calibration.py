import os
from glob import glob
from osgeo import gdal, osr
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import product_info as pinfo
from gdal_python_functions import lat_lon_pixel


def calibrate(img, sigma):
    img_cal = np.float64(img) ** 2 / sigma
    return img_cal


def denoise(img_cal, noise, mask_val):
    # Interpolate noise floor across image.
    old_idx = np.arange(0, len(noise))
    new_idx = np.linspace(0, len(noise) - 1, img_cal.shape[1])
    spl = UnivariateSpline(old_idx, noise, k=3, s=0)
    noise_ext = spl(new_idx)

    # path_save = './Dataset/Raw_SAR/' \
    #             'RS2_OK97939_PK857845_DK789666_SCWA_20180602_123655_HH_HV_SGF'
    # Visualize the extended noise signal obtained through interpolation.
    # plt.plot(noise)
    # plt.title('Sigma Nought Noise Level Values from product.xml')
    # plt.xlabel('x indices')
    # plt.ylabel('dB')
    # plt.savefig(os.path.join(path_save, 'noise.png'), bbox_inches='tight')
    # plt.show()

    # Noise is given in dB. Must first convert it to linear scale to match
    # the linear scale of the image.
    power_ref = 1
    noise_ext_lin = 10 ** (noise_ext / 10) * power_ref

    # Visualize noise levels on linear scale.
    # plt.plot(noise_ext_lin)
    # plt.title('Extended Sigma Nought Noise Level Values from product.xml')
    # plt.xlabel('x indices')
    # plt.ylabel('W')
    # plt.savefig(os.path.join(path_save, 'noise_extended.png'),
    #             bbox_inches='tight')
    # plt.show()

    # Remove noise floor from SAR image by broadcasting pixel noise values for
    # each row in image.
    img_masked = np.ma.masked_where(img_cal == mask_val, img_cal)
    mask_denoised = img_masked - noise_ext_lin
    img_denoised = mask_denoised.data
    img_denoised[np.where(img_denoised < 0)] = 0
    return img_denoised


def denoise_crop(crop, crop_center_idx, noise, raw_img_w, mask_val):
    # Interpolate noise floor across image.
    old_idx = np.arange(0, len(noise))
    new_idx = np.linspace(0, len(noise) - 1, raw_img_w)
    spl = UnivariateSpline(old_idx, noise, k=3, s=0)
    noise_ext = spl(new_idx)

    # Noise is given in dB. Must first convert it to linear scale to match
    # the linear scale of the image.
    power_ref = 1
    noise_ext_lin = 10 ** (noise_ext / 10) * power_ref

    # Crop noise relative to position of the crop in the original image it
    # was taken from.
    channels, rows, cols = crop.shape
    hw = int(cols / 2)
    noise_crop = noise_ext_lin[crop_center_idx - hw: crop_center_idx + hw]

    # Remove noise floor from SAR image by broadcasting pixel noise values for
    # each row in image.
    crop_masked = np.ma.masked_where(crop == mask_val, crop)
    mask_denoised = crop_masked - noise_crop
    crop_denoised = mask_denoised.data
    crop_denoised[np.where(crop_denoised < 0)] = 0
    return crop_denoised


def calc_angle(beta, sigma, out_shape):
    # Calculate incidence angle.
    angle = np.arcsin((beta / sigma))
    angle_mat = np.ones(out_shape)
    angle_mat = angle_mat * angle
    return angle_mat


def create_crop_GCPs(tx, affine_coeffs, crop_shape,
                     x_off, y_off, gheight, n_divs):
    """
    Create new set of Ground Control Points to properly render small crops.
    :param tx: GDAL Transformer object
    :param affine_coeffs: GDAL transform coefficients
    :param crop_shape: Shape of crop
    :param x_off: Offset from upper left corner in x direction where crop is
        located
    :param y_off: Offset from upper left corner in y direction where crop is
        located
    :param gheight: Geographical height to set in new GCPs
    :param n_divs: Number of divisions for new grid of GCPs
    :return: List of GCPs
    """
    cheight, cwidth = crop_shape

    # Make pixel coordinate grid for crop
    grid = np.mgrid[0:cwidth:n_divs*1j, 0:cheight:n_divs*1j]
    grid_crop = grid.reshape(2, -1, order='F').T

    # Calculate pixel coordinate grid of crop in the image crop is taken from.
    grid_in_og_img = np.copy(grid_crop)
    grid_in_og_img[:, 0] = grid_in_og_img[:, 0] + x_off
    grid_in_og_img[:, 1] = grid_in_og_img[:, 1] + y_off

    if tx:
        # TransformerPoints: arg[0] = bDstToSrc. 0 for SrcToDst, 1 for DstToSrc.
        # Typically used to convert image coordinates between unwarped image
        # coordinates (given by Rational Function) to image coordinates
        # in a warped image used for viewing in GIS application.
        proj_pixel_coords, success = tx.TransformPoints(0, grid_in_og_img)
    else:
        proj_pixel_coords = grid_in_og_img
    gcps = []
    for proj_coords, coords in zip(proj_pixel_coords, grid_crop):
        x_geo, y_geo = lat_lon_pixel(affine_coeffs, *proj_coords[:2])
        gcps.append(gdal.GCP(x_geo, y_geo, gheight, *coords))
    return gcps


def test_calibration_with_GDAL_bindings(path, geo_ext, rf, img):
    """
    Test GDAL methods to write calibrated/denoised image to a new TIFF.
    :param path: Directory containing SAR image to perform tests on.
    :param geo_ext: Geographical extents of the SAR image provided in the
        accompanying product.xml
    :param rf: Rational Function object
    :param img: Calibrated/denoised image
    :return: Nothing
    """
    path_sar = os.path.join(path, 'imagery_HH.tif')
    date = '_'.join(path.split('/')[-1].split('_')[-5:-3])
    height, width = img.shape
    geo_ext = np.array(geo_ext)
    geo_min = np.min(geo_ext, axis=0)
    geo_max = np.max(geo_ext, axis=0)

    # Invoke system call to gdalwarp
    te_flag = f'{geo_min[0]} {geo_min[1]} {geo_max[0]} {geo_max[1]}'
    bash_cmd = f'gdalwarp -t_srs EPSG:4326 -te {te_flag} ' \
               f'-ts {width} {height} ' \
               f'{path_sar} ' \
               f'{os.path.join(path, f"{date}_original_geocoded_CLI.tif")}'
    os.system(bash_cmd)

    # GDAL Python bindings can be used instead of using os.system()
    options = gdal.WarpOptions(
        outputBounds=(*geo_min, *geo_max),
        width=width,
        height=height,
        dstSRS='EPSG:4326'
    )
    ds_warp_src = gdal.Warp(
        os.path.join(path, f'{date}_original_geocoded_py.tif'),
        path_sar,
        options=options
    )
    ds_warp_src.FlushCache()  # Save to disk.
    # ds_warp_src = None  # Close file.

    # Get metadata needed from source file at path_sar
    ds_src = gdal.Open(path_sar)
    src_projection = ds_src.GetProjection()
    src_srs = osr.SpatialReference()  # Create empty OSRSpatialReferenceShadow.
    src_srs.ImportFromWkt(src_projection)  # Well-Known Text markup language.
    dst_srs = src_srs.ExportToWkt()  # A somewhat redundant step.

    # Create new GDAL dataset for new img.
    driver = gdal.GetDriverByName('GTiff')
    ds_dst = driver.Create(
        os.path.join(path, f'{date}_calibrated_denoised.tif'),
        width, height, 1, gdal.GDT_Float32)
    ds_dst.SetGCPs(ds_src.GetGCPs(),
                   src_srs.ExportToWkt())  # Set Ground Control Points.
    ds_dst.SetProjection(dst_srs)
    ds_dst.GetRasterBand(1).WriteArray(img)
    ds_dst.FlushCache()  # Save to disk.

    # Reproject image
    options = gdal.WarpOptions(
        outputBounds=(*geo_min, *geo_max),
        width=width,
        height=height,
        dstSRS='EPSG:4326',
        outputType=gdal.GDT_Float32
    )
    ds_warp_dst = gdal.Warp(
        os.path.join(path, f'{date}_calibrated_denoised_geocoded.tif'),
        ds_dst,
        options=options)
    ds_warp_dst.FlushCache()  # Save to disk.
    # ds_warp_dst = None  # Close file.

    # GDAL Translate to crop
    x, y = rf.lat_lon_index(6.285397691540251e+01,
                            -8.198385403380028e+01,
                            7.645005798339844e+01)
    options = gdal.TranslateOptions(srcWin=[x - 50, y - 50, 100, 100])
    ds_crop = gdal.Translate(
        os.path.join(path, f'{date}_original_crop.tif'),
        ds_src,
        options=options)
    ds_crop.FlushCache()  # Save to disk.

    def color_bar(mappable):
        ax = mappable.axes
        map_fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        return map_fig.colorbar(mappable, cax=cax)

    # Get tick labels
    xticks, xlabels, yticks, ylabels = [], [], [], []
    tx = gdal.Transformer(ds_src, ds_warp_src, ['METHOD=GCP_TPS'])
    for gcp in ds_src.GetGCPs():
        xticks.append(gcp.GCPPixel)
        xlabels.append(gcp.GCPX)
        yticks.append(gcp.GCPLine)
        ylabels.append(gcp.GCPY)
    pixel_coords = np.column_stack((xticks, yticks))
    proj_pixel_coords, success = tx.TransformPoints(0, pixel_coords)
    proj_xticks = np.array([coord[0] for coord in proj_pixel_coords])
    proj_yticks = np.array([coord[1] for coord in proj_pixel_coords])

    xlabels_sort_idx = np.argsort(xlabels)
    ylabels_sort_idx = np.argsort(ylabels)

    xticks = proj_xticks[xlabels_sort_idx]
    yticks = proj_yticks[ylabels_sort_idx]
    xlabels = np.round(xlabels, 2)[xlabels_sort_idx]
    ylabels = np.round(ylabels, 2)[ylabels_sort_idx]
    tick_idx = np.round(np.linspace(0, len(xticks) - 1, 5)).astype(int)

    # Show Results
    img1 = ds_src.GetRasterBand(1).ReadAsArray()
    img2 = ds_warp_src.GetRasterBand(1).ReadAsArray()
    img3 = ds_dst.GetRasterBand(1).ReadAsArray()
    img4 = ds_warp_dst.GetRasterBand(1).ReadAsArray()
    ax1 = plt.imshow(img1)
    color_bar(ax1)
    plt.savefig(os.path.join(path, f'{date}_original.png'))
    plt.show()
    # Set proper lat/lon coords for figure used in MDPI paper
    fig, ax2 = plt.subplots()
    tmp = ax2.imshow(img2)
    ax2.set_xticks(xticks[tick_idx])
    ax2.set_xticklabels(xlabels[tick_idx])
    ax2.set_xlabel('Lon')
    ax2.set_yticks(yticks[tick_idx])
    ax2.set_yticklabels(ylabels[tick_idx])
    ax2.set_ylabel('Lat')
    color_bar(tmp)
    plt.savefig(os.path.join(path, f'{date}_original_geocoded.png'))
    plt.show()
    ax3 = plt.imshow(img3)
    color_bar(ax3)
    plt.savefig(os.path.join(path, f'{date}_calibrated_denoised.png'))
    plt.show()
    ax4 = plt.imshow(img4)
    color_bar(ax4)
    plt.savefig(os.path.join(path, f'{date}_calibrated_denoised_geocoded.png'))
    plt.show()
    return


def test_cropping_with_GDAL_bindings(path, geo_ext):
    """
    Tests odd behavior I was seeing using GDAL. The output of gdal.Translate()
    is variably misaligned and/or cutoff when overlayed on the input image in
    QGIS depending on the size of the crop. It was found that careful
    treatment of the Ground Control Points (GCP) and using gdal.Transformer()
    with the 'METHOD=GCP_TPS' led to the closest alignment between the crop
    and the original image.
    :param path: Directory containing SAR image to perform tests on
    :param geo_ext: Geographical extents of the SAR image provided in the
        accompanying product.xml
    :return: Nothing
    """
    path_sar = os.path.join(path, 'imagery_HH.tif')
    gheight = 76.4500579833984  # geo-height

    # Get metadata needed from source file at path_sar
    ds_src = gdal.Open(path_sar)
    src_projection = ds_src.GetProjection()
    src_srs = osr.SpatialReference()  # Create empty OSRSpatialReferenceShadow.
    src_srs.ImportFromWkt(src_projection)  # Well-Known Text markup language.
    img_src = ds_src.GetRasterBand(1).ReadAsArray()
    height, width = img_src.shape

    # Warp source image to get geo-transform of warped result.
    geo_ext = np.array(geo_ext)
    geo_min = np.min(geo_ext, axis=0)
    geo_max = np.max(geo_ext, axis=0)
    options = gdal.WarpOptions(
        format='VRT',
        outputBounds=(*geo_min, *geo_max),
        width=width,
        height=height,
        dstSRS='EPSG:4326'
    )
    ds_src_warp = gdal.Warp(
        '',
        path_sar,
        options=options
    )
    # AutoCreateWarpedVRT does not give quite the same geo transform as above.
    # ds_src_warp = gdal.AutoCreateWarpedVRT(ds_src, None, dst_srs)

    ############################################################################
    # Using GDAL Translate normally for a large crop.
    ############################################################################
    x_off = 1000
    y_off = 1000
    # cheight, cwidth = 8037, 8525  # Used for:
    # RS2_OK97940_PK857859_DK789680_SCWA_20180603_120538_HH_HV_SGF
    cheight, cwidth = 7037, 8525
    img_src_crop = img_src[y_off:y_off+cheight, x_off:x_off+cwidth]
    assert (cheight, cwidth) == img_src_crop.shape

    options = gdal.TranslateOptions(srcWin=[x_off, y_off, cwidth, cheight])
    ds_crop = gdal.Translate(
        os.path.join(path, 'test_crop_og_GCPs_large.tif'),
        ds_src,
        options=options)
    ds_crop.FlushCache()
    img_dst_crop = ds_crop.GetRasterBand(1).ReadAsArray()
    assert np.sum(img_src_crop == img_dst_crop) == img_src_crop.size

    ############################################################################
    # Using GDAL Translate normally for a small crop.
    ############################################################################
    x_off = 1000
    y_off = 1000
    cheight, cwidth = 100, 100
    img_src_crop = img_src[y_off:y_off+cheight, x_off:x_off+cwidth]
    assert (cheight, cwidth) == img_src_crop.shape

    options = gdal.TranslateOptions(srcWin=[x_off, y_off, cwidth, cheight])
    ds_crop = gdal.Translate(
        os.path.join(path, 'test_crop_og_GCPs_small.tif'),
        ds_src,
        options=options)
    ds_crop.FlushCache()
    img_dst_crop = ds_crop.GetRasterBand(1).ReadAsArray()
    assert np.sum(img_src_crop == img_dst_crop) == img_src_crop.size

    ############################################################################
    # Using GDAL Translate with GCPs calculated from the source's geotransform
    # on a large crop.
    ############################################################################
    x_off = 1000
    y_off = 1000
    # cheight, cwidth = 8037, 8525
    cheight, cwidth = 7037, 8525
    img_src_crop = img_src[y_off:y_off+cheight, x_off:x_off+cwidth]
    assert (cheight, cwidth) == img_src_crop.shape

    geo_transform = gdal.GCPsToGeoTransform(ds_src.GetGCPs())
    gcps = create_crop_GCPs(None, geo_transform, img_src_crop.shape,
                            x_off, y_off, gheight, 2)
    options = gdal.TranslateOptions(srcWin=[x_off, y_off, cwidth, cheight],
                                    GCPs=gcps)
    ds_crop = gdal.Translate(
        os.path.join(path, 'test_crop_og_calculated_GCPs_large.tif'),
        ds_src,
        options=options)
    ds_crop.FlushCache()
    img_dst_crop = ds_crop.GetRasterBand(1).ReadAsArray()
    assert np.sum(img_src_crop == img_dst_crop) == img_src_crop.size

    ############################################################################
    # Using GDAL Translate with GCPs calculated from the source's geotransform
    # on a small crop.
    ############################################################################
    x_off = 1000
    y_off = 1000
    cheight, cwidth = 100, 100
    img_src_crop = img_src[y_off:y_off+cheight, x_off:x_off+cwidth]
    assert (cheight, cwidth) == img_src_crop.shape

    geo_transform = gdal.GCPsToGeoTransform(ds_src.GetGCPs())
    gcps = create_crop_GCPs(None, geo_transform, img_src_crop.shape,
                            x_off, y_off, gheight, 2)
    options = gdal.TranslateOptions(srcWin=[x_off, y_off, cwidth, cheight],
                                    GCPs=gcps)
    ds_crop = gdal.Translate(
        os.path.join(path, 'test_crop_og_calculated_GCPs_small.tif'),
        ds_src,
        options=options)
    ds_crop.FlushCache()
    img_dst_crop = ds_crop.GetRasterBand(1).ReadAsArray()
    assert np.sum(img_src_crop == img_dst_crop) == img_src_crop.size

    ############################################################################
    # Using GDAL Translate with GCPs calculated from the source's warped
    # geotransform on a large crop.
    ############################################################################
    x_off = 1000
    y_off = 1000
    # cheight, cwidth = 8037, 8525
    cheight, cwidth = 7037, 8525
    img_src_crop = img_src[y_off:y_off+cheight, x_off:x_off+cwidth]
    assert (cheight, cwidth) == img_src_crop.shape

    geo_transform = ds_src_warp.GetGeoTransform()
    gcps = create_crop_GCPs(None, geo_transform, img_src_crop.shape,
                            x_off, y_off, gheight, 2)
    options = gdal.TranslateOptions(srcWin=[x_off, y_off, cwidth, cheight],
                                    GCPs=gcps)
    ds_crop = gdal.Translate(
        os.path.join(path, 'test_crop_warp_calculated_GCPs_large.tif'),
        ds_src,
        options=options)
    ds_crop.FlushCache()
    img_dst_crop = ds_crop.GetRasterBand(1).ReadAsArray()
    assert np.sum(img_src_crop == img_dst_crop) == img_src_crop.size

    ############################################################################
    # Using GDAL Translate with GCPs calculated from the source's warped
    # geotransform on a small crop.
    ############################################################################
    x_off = 1000
    y_off = 1000
    cheight, cwidth = 100, 100
    img_src_crop = img_src[y_off:y_off+cheight, x_off:x_off+cwidth]
    assert (cheight, cwidth) == img_src_crop.shape

    geo_transform = ds_src_warp.GetGeoTransform()
    gcps = create_crop_GCPs(None, geo_transform, img_src_crop.shape,
                            x_off, y_off, gheight, 2)
    options = gdal.TranslateOptions(srcWin=[x_off, y_off, cwidth, cheight],
                                    GCPs=gcps)
    ds_crop = gdal.Translate(
        os.path.join(path, 'test_crop_warp_calculated_GCPs_small.tif'),
        ds_src,
        options=options)
    ds_crop.FlushCache()
    img_dst_crop = ds_crop.GetRasterBand(1).ReadAsArray()
    assert np.sum(img_src_crop == img_dst_crop) == img_src_crop.size

    ############################################################################
    # Method that gives closest results to desired outcome.
    ############################################################################
    geo_transform = ds_src_warp.GetGeoTransform()
    # inv = gdal.InvGeoTransform(geo_transform)

    x_off = 1000
    y_off = 1000
    # cheight, cwidth = 8037, 8525
    cheight, cwidth = 7037, 8525
    img_src_crop = img_src[y_off:y_off+cheight, x_off:x_off+cwidth]
    assert (cheight, cwidth) == img_src_crop.shape

    tx = gdal.Transformer(ds_src, ds_src_warp, ['METHOD=GCP_TPS'])
    gcps = create_crop_GCPs(tx, geo_transform, img_src_crop.shape,
                            x_off, y_off, gheight, 10)
    options = gdal.TranslateOptions(srcWin=[x_off, y_off, cwidth, cheight],
                                    GCPs=gcps)
    ds_crop = gdal.Translate(
        os.path.join(path, 'test_crop_correct_GCPs_large.tif'),
        ds_src,
        options=options)
    ds_crop.FlushCache()
    img_dst_crop = ds_crop.GetRasterBand(1).ReadAsArray()
    assert np.sum(img_src_crop == img_dst_crop) == img_src_crop.size

    ############################################################################
    # Repeat for smaller crop to get closest results to desired outcome.
    ############################################################################
    x_off = 1000
    y_off = 1000
    cheight, cwidth = 100, 100
    img_src_crop = img_src[y_off:y_off+cheight, x_off:x_off+cwidth]
    assert (cheight, cwidth) == img_src_crop.shape

    tx = gdal.Transformer(ds_src, ds_src_warp, ['METHOD=GCP_TPS'])
    gcps = create_crop_GCPs(tx, geo_transform, img_src_crop.shape,
                            x_off, y_off, gheight, 10)
    options = gdal.TranslateOptions(srcWin=[x_off, y_off, cwidth, cheight],
                                    GCPs=gcps)
    ds_crop = gdal.Translate(
        os.path.join(path, 'test_crop_correct_GCPs_small.tif'),
        ds_src,
        options=options)
    ds_crop.FlushCache()
    img_dst_crop = ds_crop.GetRasterBand(1).ReadAsArray()
    assert np.sum(img_src_crop == img_dst_crop) == img_src_crop.size


def main():
    """
    Test the functions included in this file for a particular SAR image.
    """
    IMG_ROOT = 'RS2_OK97939_PK857845_DK789666_SCWA_20180602_123655'
    PATH_RAW = glob(os.path.join('./Dataset/Raw_SAR', IMG_ROOT + '*'))[0]
    PATH_SAR = os.path.join(PATH_RAW, 'imagery_HH.tif')
    PATH_PROD = os.path.join(PATH_RAW, 'product.xml')
    PATH_BETA = os.path.join(PATH_RAW, 'lutBeta.xml')
    PATH_SIGMA = os.path.join(PATH_RAW, 'lutSigma.xml')
    SAR_FILL_VALUE = 0

    ds_sar = gdal.Open(PATH_SAR)
    img = ds_sar.GetRasterBand(1).ReadAsArray()

    # Get metadata from XML files.
    noise = pinfo.get_noise(PATH_PROD)
    beta = pinfo.get_look_up_table(PATH_BETA)
    sigma = pinfo.get_look_up_table(PATH_SIGMA)
    geo_ext = pinfo.get_extents(PATH_PROD, *img.shape)
    rf = pinfo.get_rational_function(PATH_PROD)

    # Visualize sigma
    plt.plot(sigma)
    # plt.title('Sigma Gain Values from lutSigma.xml')
    plt.xlabel('x indices')
    plt.ylabel('Gain')
    plt.savefig(os.path.join(os.path.join(PATH_RAW, 'sigma.png')),
                bbox_inches='tight')
    plt.show()

    # Calibrate image
    img_cal = calibrate(img, sigma)
    # Remove noise from calibrated
    img_denoised = denoise(img_cal, noise, SAR_FILL_VALUE)
    # Calculate incidence angle across
    img_angle = calc_angle(beta, sigma, img.shape)

    test_calibration_with_GDAL_bindings(PATH_RAW, geo_ext, rf, img_denoised)
    test_cropping_with_GDAL_bindings(PATH_RAW, geo_ext)

    return img_cal, img_denoised, img_angle


if __name__ == '__main__':
    main()
    print('Program Terminated')
