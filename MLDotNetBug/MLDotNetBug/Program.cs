using Microsoft.ML.Transforms;
using Microsoft.ML;
using MLDotNetBug.Utils;
using Microsoft.ML.Vision;

var datasetPath = PathHelper.GetAbsolutePath(@"../../../data");

var mlContext = new MLContext();
mlContext.Log += (sender, eventArgs) => Console.WriteLine(eventArgs.Message);


var dataset = DatasetLoader.LoadImages(datasetPath);
var splitDataset = mlContext.Data.TrainTestSplit(dataset, 0.2);


//train

var preprocessingPipeline = mlContext.Transforms.Conversion
    .MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelAsKey", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
    .Append(mlContext.Transforms.LoadRawImageBytes(imageFolder: datasetPath, inputColumnName: "ImagePath", outputColumnName: "Feature"));

var preprocessedData = preprocessingPipeline.Fit(splitDataset.TrainSet).Transform(splitDataset.TrainSet);
var validationTestSplit = mlContext.Data.TrainTestSplit(preprocessedData, testFraction: 0.5);
var trainSet = validationTestSplit.TrainSet;


var trainingOptions = new ImageClassificationTrainer.Options() { FeatureColumnName = "Feature", LabelColumnName = "LabelAsKey", Arch = ImageClassificationTrainer.Architecture.MobilenetV2};
var trainingPipeline = mlContext.MulticlassClassification.Trainers
.ImageClassification(trainingOptions)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
    .AppendCacheCheckpoint(mlContext);

var trainedModel = preprocessingPipeline.Append(trainingPipeline).Fit(trainSet);

Console.WriteLine("Successfully trained.");