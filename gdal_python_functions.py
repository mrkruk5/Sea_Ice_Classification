"""
Unused file that was copied from another project that I thought may be
useful. Turned out that none of these functions were needed due to
information provided in auxiliary files accompanying the SAR images. These
provided geographical extents as well as Rational Function coefficents to
convert between projection coordinates and image indices.
"""
from osgeo import gdal, osr
from collections import namedtuple


def lat_lon_pixel(gt, xInd, yInd):
    x = gt[0] + xInd*gt[1] + yInd*gt[2]
    y = gt[3] + xInd*gt[4] + yInd*gt[5]
    return [x, y]


def lat_lon_index(gt, lat, lon):
    xInd = (((lon - gt[0]) / gt[1]) + ((-lat * gt[2] + gt[2] * gt[3]) / gt[1] / gt[5])) / \
           (1 - (gt[2] * gt[4] / gt[1] / gt[5]))
    yInd = (((lat - gt[3]) / gt[5]) + ((-lon * gt[4] + gt[0] * gt[4]) / gt[1] / gt[5])) / \
           (1 - (gt[2] * gt[4] / gt[1] / gt[5]))
    return [int(xInd), int(yInd)]


def GetExtent(gt, cols, rows):
    ext = []
    xarr = [0, cols]
    yarr = [0, rows]

    for px in xarr:
        for py in yarr:
            x = gt[0] + (px*gt[1]) + (py*gt[2])
            y = gt[3] + (px*gt[4]) + (py*gt[5])
            ext.append([x, y])
    return ext


def ReprojectCoords(coords, src_srs, tgt_srs):
    trans_coords = []
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x, y in coords:
        x, y, z = transform.TransformPoint(x, y)
        trans_coords.append([x, y])
    return trans_coords


def latLonDMS(x, y):
    [dx, mx, sx] = deg_to_dms(x)
    if dx > 0:
        strx = f"""{dx}d{mx}'{sx:.2f}\"E"""
    else:
        strx = f"""{abs(dx)}d{mx}'{sx:.2f}\"W"""
    [dy, my, sy] = deg_to_dms(y)
    if dy > 0:
        stry = f"""{dy}d{my}'{sy:.2f}\"N"""
    else:
        stry = f"""{dy}d{my}'{sy:.2f}\"S"""
    return [strx, stry]


def deg_to_dms(deg):
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]


def printFileMeta(ds):
    print("Metadata:")
    meta = ds.GetMetadata_List()  # dataset.GetMetadata_Dict() returns same data but to a dict
    for i in meta:
        print(f"  {i}")


def printImageMeta(ds):
    print("Image Structure Metadata:")
    meta = ds.GetMetadata('IMAGE_STRUCTURE')
    for i in meta:
        print(f"  {i}")


def printCoordinates(list):
    dms = []
    coordNames = ["Upper Left", "Lower Left", "Upper Right", "Lower Right"]
    print("Corner Coordinates:")
    for i in list:
        dms.append(latLonDMS(i[0], i[1]))
    for j in range(len(list)):
        print(f"{coordNames[j]:<11} ({list[j][0]:.7f}, {list[j][1]:.7f}) ({dms[j][0]}, {dms[j][1]})")


def printStats(statTitles, stats):
    min, max, mean, stdDev = stats
    stats = [max, mean, min, stdDev]  # Reorder to match gdalinfo
    print("  Metadata:")
    for i in range(len(stats)):
        print(f"\t{statTitles[i]}={stats[i]}")


def get_data(file):
    # GetGeoTransform() returns 6 coefficients (GT) to perform the Affine transform that relates raster pixels to
    # georeferenced coordinates.
    # topLeftX, pixWidth, gt2, topLeftY, gt4, pixHeight = geotransform
    # In north up images, GT(2) and GT(4) coefficients are zero.
    # Affine transform:
    # Xgeo = GT(0) + Xpixel * GT(1) + Yline * GT(2)
    # Ygeo = GT(3) + Xpixel * GT(4) + Yline * GT(5)
    # Note: The indices refer to the names in the titles variable.

    #
    # Data Extraction
    #
    dataset = gdal.Open(file, gdal.GA_ReadOnly)
    projection = dataset.GetProjection()  # gives SRS in WKT
    geotransform = dataset.GetGeoTransform()
    ext = GetExtent(geotransform, dataset.RasterXSize, dataset.RasterYSize)
    src_srs = osr.SpatialReference()  # Makes an empty SpatialReference object
    src_srs.ImportFromWkt(projection)  # Populates the SpatialReference object with our WKT SRS
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)
    # tgt_srs = src_srs.CloneGeogCS()
    geo_ext = ReprojectCoords(ext, src_srs, tgt_srs)
    band = dataset.GetRasterBand(1)
    # nodata = band.GetNoDataValue()
    stats = band.GetStatistics(0, 1)

    Data = namedtuple('Data', ['dataset', 'projection', 'geotransform', 'src_srs', 'geo_ext', 'band', 'stats'])
    return Data(dataset, projection, geotransform, src_srs, geo_ext, band, stats)


def gdalinfo_python(data):
    #
    # Print data in the same format as gdalinfo
    #
    print(f"Driver: {data.dataset.GetDriver().ShortName}/{data.dataset.GetDriver().LongName}")
    print(f"Size is {data.dataset.RasterXSize} x {data.dataset.RasterYSize} x {data.dataset.RasterCount}")
    print(f"Coordinate System is:\n{data.src_srs}")
    print(f"Origin = ({data.geotransform[0]}, {data.geotransform[3]})")
    print(f"Pixel Size = ({data.geotransform[1]}, {data.geotransform[5]})")
    printFileMeta(data.dataset)
    printImageMeta(data.dataset)
    printCoordinates(data.geo_ext)
    print(f"Band {data.dataset.RasterCount} Type={gdal.GetDataTypeName(data.band.DataType)}")
    min = data.band.GetMinimum()
    max = data.band.GetMaximum()
    if not min or not max:
        (min, max) = data.band.ComputeRasterMinMax(True)
    print(f"Min={min:.3f} Max={max:.3f}")
    min, max, mean, stdDev = data.stats
    print(f"Minimum={min}, Maximum={max}, Mean={mean}, StdDev={stdDev}")
    statTitles = ["STATISTICS_MAXIMUM", "STATISTICS_MINIMUM", "STATISTICS_MEAN", "STATISTICS_STDDEV"]
    printStats(statTitles, data.stats)