rng('default')

disp('Creating train and validation datastores...')
sampleSize = 12560;
[fdsTrainCombined, fdsValCombined] = createTrainValidationDatastores(sampleSize);


% Create and train a network
options = trainingOptions(...
    'adam',...
    'InitialLearnRate',0.0005,...
    'ValidationData', fdsValCombined,...
    'MiniBatchSize', 64, ...
    'MaxEpochs', 9, ...
    'Plots','none', ...
    'Verbose',true);

numWavelengths = 55;
numFeatures = 300;


layers = [
    sequenceInputLayer(numFeatures)
    fullyConnectedLayer(1024)
    reluLayer()
    fullyConnectedLayer(256)
    reluLayer()
    fullyConnectedLayer(1)
    regressionLayer
    ];


net = trainNetwork(fdsTrainCombined,layers,options);
%save('baseline_trainedNet','net')

disp('Computing predictions on validation set...')

% Do some predictions
targs = readall(fdsValCombined.UnderlyingDatastores{2});
predsVal = predict(net, fdsValCombined);

disp('Computing Ariel score on validation set...')
predsVal = predsVal';
score = arielMetric(predsVal, targs)










