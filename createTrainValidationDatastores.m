function [fdsTrainCombined, fdsValCombined] = createTrainValidationDatastores(sampleSize)


c = split(fileread('noisy_train.txt'));
c = cellfun(@(x) replace(x, 'noisy_train', ''), c, 'UniformOutput', false);
n = numel(c);
ii = randperm(n);
listLightCurves = c(ii);

listLightCurves = listLightCurves(1:sampleSize);

partition = cvpartition(sampleSize,'HoldOut',0.2);
uniqueIds = 1:sampleSize;

trainIds = uniqueIds(partition.training);
valIds = uniqueIds(partition.test);

trainFiles = listLightCurves(trainIds);
valFiles = listLightCurves(valIds);


train_folderName = "/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train";
trainPredictors = convertCharsToStrings(cellstr(cellfun(@(x) append(train_folderName, x), trainFiles, 'UniformOutput', false)));
valPredictors = convertCharsToStrings(cellstr(cellfun(@(x) append(train_folderName, x), valFiles, 'UniformOutput', false)));

params_folderName = '/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train';
trainTargets = convertCharsToStrings(cellstr(cellfun(@(x) append(params_folderName, x), trainFiles, 'UniformOutput', false)));
valTargets = convertCharsToStrings(cellstr(cellfun(@(x) append(params_folderName, x), valFiles, 'UniformOutput', false)));

% Train
fdsTrainPredictors = fileDatastore(trainPredictors, ...
    'ReadFcn',@(filename) handlePredictors(filename));

fdsTrainTargets = fileDatastore(trainTargets, ...
    'ReadFcn',@(filename) handleTargets(filename));

fdsTrainCombined = combine(fdsTrainPredictors, fdsTrainTargets);

% Validation
fdsValPredictors = fileDatastore(valPredictors, ...
    'ReadFcn',@(filename) handlePredictors(filename));

fdsValTargets = fileDatastore(valTargets, ...
    'ReadFcn',@(filename) handleTargets(filename));

fdsValCombined = combine(fdsValPredictors, fdsValTargets);

end


function lightcurve = handlePredictors(filename)
  lightcurve = readmatrix(filename, 'Range', 7);
  lightcurve = lightcurve';
  lightcurve = (lightcurve - 1) ./ 0.04;
end

function target = handleTargets(filename)
  target = readmatrix(filename, 'Range', 3);
end


