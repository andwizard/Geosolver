import os
import json
from typing import  Dict, Any, Optional, Annotated
import geopandas as gpd
import pandas as pd
from osgeo import gdal
import rasterio
import shapefile
from shapely.validation import explain_validity
from langchain_core.tools import tool
from utils.logger import get_logger_with_session
import warnings
warnings.filterwarnings("ignore")

@tool
def read_shapefile_info(
    file_name: Annotated[str, "Name of the shapefile to read"]
) -> Dict[str, Any]:
    """
    Read the basic information of a specified Shapefile.

    This function extracts key metadata from a Shapefile, including its coordinate system,
    geometry type, number of features, and attribute fields. It is particularly useful 
    for understanding the structure of a Shapefile before generating code for further processing.

    Args:
        file_name (str): The name of the Shapefile to read (without the .shp extension).

    Returns:
        dict: A dictionary containing the Shapefile's metadata or an error message if the 
        operation fails. The returned dictionary includes:
    """
    logger = get_logger_with_session()
    try:
        working_directory = os.environ["WORKING_DIRECTORY"]
        if working_directory not in file_name:
            file_path = os.path.join(working_directory, file_name + ".shp")
        
        if not os.path.exists(file_path):
            return {"error": f"Error: The file {file_name} was not found."}
        
        sf = shapefile.Reader(file_path)
        shape_type = sf.shapeTypeName
        num_records = len(sf)      
        fields = sf.fields[1:]        
        
        field_info = [{"name": field[0], "type": field[1]} for field in fields]
        
        prj_path = file_path.replace(".shp", ".prj")
        if os.path.exists(prj_path):
            with open(prj_path, "r", encoding="utf-8") as prj_file:
                prj_info = prj_file.read()
        else:
            prj_info = "No projection information available."
        
        shapefile_info = {
            "file_name": file_name,
            "geometry_type": shape_type,
            "num_features": num_records,
            "fields": field_info,
            "projection": prj_info
        }
        return shapefile_info
    
    except shapefile.ShapefileException as e:
        logger.error(f"Shapefile reading error: {str(e)}")
        return {"error": f"Shapefile reading error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error while reading shapefile: {str(e)}")
        return {"error": f"Unexpected error while reading shapefile: {str(e)}"}
    
@tool
def reading_gis_metadata(
    file_name: Annotated[str, "Path to the input GIS file"]
) -> str:
    """
    Generate metadata for a GIS file (vector or raster) and return it as a JSON-formatted string.

    This function reads the input GIS file and generates metadata about its contents. It does not process table files like CSV or Excel.

    Args:
        file_name (str): The path to the GIS file (e.g., .shp, .tif). The file must be in the
        working directory or its full path must be provided.

    Returns:
        str: A JSON-formatted string containing the metadata. If an error occurs, the JSON will
        include an "error" field with the error message.

    Notes:
        - For vector data, the function calculates geometric statistics (centroid and spatial variance) 
          and attribute table details.
        - For raster data, it extracts band statistics, resolution, dimensions, and no-data values.
        - Table data such as CSV and Excel files are explicitly not supported.
    """

    logger = get_logger_with_session()

    def _convert_dates_in_dataframe(dataframe):
        for column in dataframe.columns:
            if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
                dataframe[column] = dataframe[column].dt.strftime('%Y-%m-%d')
        return dataframe
    
    def _calculate_geometry_stats(vector_data):
        try:
            if vector_data.crs and vector_data.crs.is_geographic:
                projected_data = vector_data.to_crs(epsg=3857)
                centroid = projected_data.geometry.centroid
                centroid_x = centroid.x.mean()
                centroid_y = centroid.y.mean()
                spatial_variance = centroid.distance(centroid[0]).mean()
                
                if vector_data.crs:
                    centroid_points = gpd.GeoSeries([centroid[0]], crs=3857).to_crs(vector_data.crs)
                    centroid_x = centroid_points.x[0]
                    centroid_y = centroid_points.y[0]
            else:
                centroid = vector_data.geometry.centroid
                centroid_x = centroid.x.mean()
                centroid_y = centroid.y.mean()
                spatial_variance = centroid.distance(centroid[0]).mean()
                
            return {
                "centroid_x": float(centroid_x),
                "centroid_y": float(centroid_y),
                "spatial_variance": float(spatial_variance)
            }
        except Exception as e:
            logger.warning(f"Error calculating geometry stats: {str(e)}")
            return {
                "centroid_x": None,
                "centroid_y": None,
                "spatial_variance": None
            }
        
    def _read_file_with_encoding(file_path, encodings=["utf-8", "ISO-8859-1"]):
        for encoding in encodings:
            try:
                return gpd.read_file(file_path, encoding=encoding), encoding
            except UnicodeDecodeError as e:
                logger.warning(f"(reading_gis_metadata) Error reading file with encoding {encoding}: {e}")
        raise Exception("Failed to read the file with provided encodings.")

    def _check_invalid_geometries(vector_data):
        invalid_geometries = vector_data[~vector_data.is_valid]
        invalid_count = len(invalid_geometries)
        invalid_details = []
        
        for geom in invalid_geometries.geometry:
            invalid_details.append(explain_validity(geom))
        
        return invalid_count, invalid_details

    working_directory = os.environ["WORKING_DIRECTORY"]
    if working_directory not in file_name:
        file_path = os.path.join(working_directory, file_name)
    
    if not os.path.exists(file_path):
        return json.dumps({"error": f"Error: The file {file_name} was not found."}, ensure_ascii=False)
    
    if file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
        return json.dumps({"error": "This function cannot process table files. Please use read_csv_data() function."}, ensure_ascii=False)

    logger.info(f"(reading_gis_metadata) Reading GIS metadata, file_name: {file_name}")

    metadata = {
        "metadata": {
            "general": None,
            "vector_data": None,
            "raster_data": None
        }
    }

    # Determine file size
    file_size = os.path.getsize(file_path)
    disk_size = f"{file_size / (1024 * 1024):.2f} MB"

    # Check the file extension and handle accordingly
    file_extension = file_path.lower().split('.')[-1]

    if file_extension in ['shp', 'geojson', 'gpkg', 'kml', 'gml']:  # Vector formats
        try:
            # Attempt to read as vector data
            vector_data, encoding_used = _read_file_with_encoding(file_path)
            # logger.info(f"(reading_gis_metadata) Successfully read the file using encoding: {encoding_used}")
            if not vector_data.empty:
                vector_data = _convert_dates_in_dataframe(vector_data)

                # Extract general metadata
                crs = vector_data.crs.to_string() if vector_data.crs else "Unknown"
                bounds = vector_data.total_bounds
                feature_count = len(vector_data)

                # Extract attribute metadata
                fields = []
                for col in vector_data.columns:
                    if col != vector_data.geometry.name:
                        sample_values = vector_data[col].dropna().unique()[:5].tolist()
                        field_type = str(vector_data[col].dtype)
                        fields.append({
                            "field_name": col,
                            "data_type": field_type,
                            "sample_values": sample_values,
                            "possible_meaning": None
                        })

                geometry_stats = _calculate_geometry_stats(vector_data)

                invalid_count, invalid_details = _check_invalid_geometries(vector_data)

                metadata["metadata"]["general"] = {
                    "file_name": os.path.basename(file_name),
                    "file_type": "vector",
                    "file_encoding": encoding_used,
                    "crs": crs,
                    "extent": {
                        "xmin": float(bounds[0]),
                        "ymin": float(bounds[1]),
                        "xmax": float(bounds[2]),
                        "ymax": float(bounds[3]),
                    },
                    "data_type": "vector",
                    "size": {
                        "disk_size": disk_size,
                        "feature_count": feature_count,
                    }
                }
                metadata["metadata"]["vector_data"] = {
                    "geometry_type": vector_data.geometry.type.unique().tolist(),
                    "attribute_table": {"fields": fields},
                    "geometry_stats": {
                        "total_features": feature_count,
                        "spatial_distribution": geometry_stats,
                        "invalid_geometries_count": invalid_count,
                        "invalid_geometries_details": invalid_details
                    }
                }

                return json.dumps(metadata, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.warning(f"(reading_gis_metadata) Error processing vector data: {str(e)}")
            return json.dumps({"error": f"Error processing vector data: {str(e)}"}, ensure_ascii=False)

    if file_extension in ['tif', 'img', 'nc']:  # Raster formats
        try:
            logger.info(f"(reading_gis_metadata) Reading raster data, file_name: {file_name}")
            with rasterio.open(file_path) as src:
                bounds = src.bounds
                crs = src.crs.to_string() if src.crs else "Unknown"
                band_stats = []
                for i in range(1, src.count + 1):
                    band = src.read(i)
                    band_stats.append({
                        "band_number": i,
                        "description": f"Band {i}",
                        "min_value": float(band.min()),
                        "max_value": float(band.max()),
                        "mean_value": float(band.mean()),
                        "data_type": str(band.dtype),
                        "possible_meaning": None
                    })
                metadata["metadata"]["general"] = {
                    "file_name": os.path.basename(file_name),
                    "file_type": "raster",
                    "crs": crs,
                    "extent": {
                        "xmin": float(bounds.left),
                        "ymin": float(bounds.bottom),
                        "xmax": float(bounds.right),
                        "ymax": float(bounds.top),
                    },
                    "data_type": "raster",
                    "size": {
                        "disk_size": disk_size,
                        "feature_count": None,
                    }
                }
                metadata["metadata"]["raster_data"] = {
                    "dimensions": {
                        "width": src.width,
                        "height": src.height,
                        "resolution": {
                            "x": float(src.res[0]),
                            "y": float(src.res[1])
                        }
                    },
                    "bands": band_stats,
                    "no_data_value": float(src.nodatavals[0]) if src.nodatavals else None,
                    "statistics": {
                        "min": float(min(band.min() for band in src.read())),
                        "max": float(max(band.max() for band in src.read())),
                        "mean": float(src.read().mean()),
                        "std_dev": float(src.read().std())
                    },
                    "possible_meaning": None
                }
                return json.dumps(metadata, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.warning(f"(reading_gis_metadata) Error processing raster data: {str(e)}")

            try:
                logger.info(f"(reading_gis_metadata) Reading raster data with GDAL, file_name: {file_name}")
                ds = gdal.Open(file_path)
                if ds is None:
                    raise Exception("GDAL failed to open the file.")

                geotransform = ds.GetGeoTransform()
                projection = ds.GetProjection()
                bands = ds.RasterCount

                band_stats = []
                min_values = []
                max_values = []
                mean_values = []
                stddev_values = []

                for i in range(1, bands + 1):
                    band = ds.GetRasterBand(i)

                    min_val = band.GetMinimum()
                    max_val = band.GetMaximum()
                    mean_val = band.GetStatistics(0, 1)[2] 
                    stddev_val = band.GetStatistics(0, 1)[3]

                    band_stats.append({
                        "band_number": i,
                        "description": f"Band {i}",
                        "min_value": float(min_val) if min_val is not None else None,
                        "max_value": float(max_val) if max_val is not None else None,
                        "mean_value": float(mean_val) if mean_val is not None else None,
                        "std_dev": float(stddev_val) if stddev_val is not None else None,
                        "data_type": str(band.DataType),
                        "possible_meaning": None
                    })

                    min_values.append(min_val)
                    max_values.append(max_val)
                    mean_values.append(mean_val)
                    stddev_values.append(stddev_val)

                overall_min = min(min_values) if min_values else None
                overall_max = max(max_values) if max_values else None
                overall_mean = sum(mean_values) / len(mean_values) if mean_values else None
                overall_stddev = sum(stddev_values) / len(stddev_values) if stddev_values else None

                metadata["metadata"]["general"] = {
                    "file_name": os.path.basename(file_name),
                    "file_type": "raster",
                    "crs": projection,
                    "extent": {
                        "xmin": float(geotransform[0]),
                        "ymin": float(geotransform[3] + geotransform[5] * ds.RasterYSize),
                        "xmax": float(geotransform[0] + geotransform[1] * ds.RasterXSize),
                        "ymax": float(geotransform[3]),
                    },
                    "data_type": "raster",
                    "size": {
                        "disk_size": disk_size,
                        "feature_count": None,
                    }
                }

                metadata["metadata"]["raster_data"] = {
                    "dimensions": {
                        "width": ds.RasterXSize,
                        "height": ds.RasterYSize,
                        "resolution": {
                            "x": float(geotransform[1]),
                            "y": float(geotransform[5])
                        }
                    },
                    "bands": band_stats,
                    "no_data_value": float(ds.GetRasterBand(1).GetNoDataValue()) if ds.GetRasterBand(1).GetNoDataValue() is not None else None,
                    "statistics": {
                        "min": float(overall_min) if overall_min is not None else None,
                        "max": float(overall_max) if overall_max is not None else None,
                        "mean": float(overall_mean) if overall_mean is not None else None,
                        "std_dev": float(overall_stddev) if overall_stddev is not None else None
                    },
                    "possible_meaning": None
                }

                return json.dumps(metadata, ensure_ascii=False, indent=4)

            except Exception as e:
                logger.warning(f"(reading_gis_metadata) Error processing raster data with GDAL: {str(e)}")
                return json.dumps({"error": str(e)}, ensure_ascii=False, indent=4)
