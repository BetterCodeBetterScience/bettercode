I would like to generate a Jupyter notebook for analysis of the NYC Taxi data.

- first we will download data from the NYC Taxi data.  the data files are of the format "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet" and I'd like to download 2015-01 through 2024-12 and save them to the nyctaxi/orig subdir within the DATADIR specified in the .env file, unless the file already exists. also download the taxi zone metadata from https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv unless it already exists in the datadir.  

- second we will load each of the data files and use the PULocationID and DOLocationID variables to create new PUBorough and DOBorough variables using the LocationID and Borough variables within the taxi zone metadata file.  resave these individual files into the nyctaxi/preproc subdir

- load all of the preproc data files and combine the the data, saving them to a single parquet file. 

- create a new duckdb database that contains the entire dataset as well

- The goal of our analysis is to determine whether the pattern of trips within/between different boroughs changes across the months of the year and over time. For this we need to determine the frequency of each combination of PUBorough and DOBorough, separately by month/year. this should be stored as a matrix with the number of within-borough trips on the diagonal, and the directed between-borough trips on the off-diagonal, with pickup as the first index and dropoff as the second.

- Once the analysis function is created, we want to assess how long it takes to apply it using three different methods:
- load each individual file and compute the pattern
- load the single parquet file and compute the pattern for each month
- use the duckdb database to compute the pattern for each month