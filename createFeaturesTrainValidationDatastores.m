function [fdsTrainCombined, fdsValCombined] = createFeaturesTrainValidationDatastores(sampleSize)
% This function requires files from the Ariel 2021 Data Challenge
% See: https://www.ariel-datachallenge.space/

c = split(fileread('noisy_train.txt'));
c = cellfun(@(x) replace(x, 'noisy_train', ''), c, 'UniformOutput', false);
n = numel(c);
ii = randperm(n);
listLightCurves = c(ii);

listLightCurves = listLightCurves(1:sampleSize);

% Split into 80/20 for training and validation.
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

fdsTrainFeatures = fileDatastore(trainPredictors, ...
    'ReadFcn',@(filename) handleFeatures(filename));

fdsTrainTargets = fileDatastore(trainTargets, ...
    'ReadFcn',@(filename) handleTargets(filename));

fdsTrainCombined = combine(fdsTrainPredictors, fdsTrainFeatures, fdsTrainTargets);

% Validation
fdsValPredictors = fileDatastore(valPredictors, ...
    'ReadFcn',@(filename) handlePredictors(filename));

fdsValFeatures = fileDatastore(valPredictors, ...
    'ReadFcn',@(filename) handleFeatures(filename));

fdsValTargets = fileDatastore(valTargets, ...
    'ReadFcn',@(filename) handleTargets(filename));

fdsValCombined = combine(fdsValPredictors, fdsValFeatures, fdsValTargets);

end


function lightcurve = handlePredictors(filename)
% Normalizes the light curve points.
lightcurve = readmatrix(filename, 'Range', 7);
lightcurve = lightcurve';
lightcurve = (lightcurve - 1) ./ 0.04;
end

function features = handleFeatures(filename)
% Loads the 6 numeric features from each files.
pat = ["# star_temp: ", "# star_logg: ", "# star_rad: ", "# star_mass: ", "# star_k_mag: ","# period: "];
a = table2cell(readtable(filename, 'ReadVariableNames', false, 'Range',[1,1,6,1]));
features = cell2mat(cellfun(@(x) str2double(replace(x, pat,'')),a,'Un',0))';
end

function target = handleTargets(filename)
% Loads the planet-to-star ratio values, which will be our targets.
target = readmatrix(filename, 'Range', 3);
end


