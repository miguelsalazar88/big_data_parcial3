from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql.functions import lit

spark = SparkSession.builder.appName('NewsProcessing').getOrCreate()
carpetas = ["s3://headlines-final/el_tiempo/*.csv", "s3://headlines-final/el_espectador/*.csv"]
data = spark.read.csv(carpetas, header=True, inferSchema=True)

tokenizer = Tokenizer(inputCol='categoria', outputCol='categoria_tokenized')
data_tokenized = tokenizer.transform(data)

tokenizer = Tokenizer(inputCol="titular", outputCol="titular_tokenized")
data_tokenized = tokenizer.transform(data_tokenized)

tokenizer = Tokenizer(inputCol="enlace", outputCol="enlace_tokenized")
data_tokenized = tokenizer.transform(data_tokenized)

hashingTF = HashingTF(inputCol="categoria_tokenized", outputCol="categoria_features", numFeatures=10000)
data_tf = hashingTF.transform(data_tokenized)

hashingTF = HashingTF(inputCol="titular_tokenized", outputCol="titular_features", numFeatures=10000)
data_tf = hashingTF.transform(data_tf)

hashingTF = HashingTF(inputCol="enlace_tokenized", outputCol="enlace_features", numFeatures=10000)
data_tf = hashingTF.transform(data_tf)

idf = IDF(inputCol="categoria_features", outputCol="categoria_tfidf")
idfModel = idf.fit(data_tf)
data_tfidf = idfModel.transform(data_tf)

idf = IDF(inputCol="titular_features", outputCol="titular_tfidf")
idfModel = idf.fit(data_tfidf)
data_tfidf = idfModel.transform(data_tfidf)

idf = IDF(inputCol="enlace_features", outputCol="enlace_tfidf")
idfModel = idf.fit(data_tfidf)
data_tfidf = idfModel.transform(data_tfidf)

data_tfidf.write.parquet('s3://spark-parcial3/resultados.parquet')