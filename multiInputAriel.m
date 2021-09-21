rng('default')

disp('Creating training and validation datastores...')

sampleSize = 125600;
[fdsTrainCombined, fdsValCombined] = createFeaturesTrainValidationDatastores(sampleSize);

disp('Setting up network...')
numTransitFeatures = 300;
numWavelengths = 55;
imageInputSize = [numTransitFeatures numWavelengths 1];
 
% First part of the network
 layers = [
    imageInputLayer(imageInputSize,'Normalization','none','Name','transits')
    convolution2dLayer(16,8,'Padding','same','Name','conv2d1')
    batchNormalizationLayer('Name','batch1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','max1') 
    
    convolution2dLayer(16,16,'Padding','same','Name','conv2d2')
    batchNormalizationLayer('Name','batch2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','max2') 
    
    convolution2dLayer(16,32,'Padding','same','Name','conv2d3')
    batchNormalizationLayer('Name','batch3')
    reluLayer('Name','relu3')
    maxPooling2dLayer(2,'Stride',2,'Name','max3') 
    
    convolution2dLayer(16,64,'Padding','same','Name','conv2d4')
    batchNormalizationLayer('Name','batch4')
    reluLayer('Name','relu4')
    maxPooling2dLayer(2,'Stride',2,'Name','max4') 
    
    fullyConnectedLayer(55,'Name','fc1')
    
    concatenationLayer(1,2,'Name','concat')
    
    fullyConnectedLayer(55,'Name','fc3')
    ];

% Convert the layers to a layer graph.
lgraph = layerGraph(layers);

% For the second part of the network, add a feature input layer and connect it
% to the second input of the concatenation layer.
numFeatures = 6;

% Load precomputed max and min values of features, for normalization.
load normalized_values.mat

layers2= [
    featureInputLayer(numFeatures,...
    'Normalization','rescale-zero-one',...
    'Min',min_values,...
    'Max',max_values,...
    'Name','features')
    fullyConnectedLayer(55,'Name','fc2')
    ];

lgraph = addLayers(lgraph, layers2);
lgraph = connectLayers(lgraph, 'fc2', 'concat/in2');
 
% Create a dlnetwork object.
dlnet = dlnetwork(lgraph);

% Train with a mini-batch size of 64.
numEpochs = 9;
miniBatchSize = 64;

% Specify the initial learning rate for Adam optimization. 
learnRate = 0.005;

plots = "training-progress";

gradientsAvg = [];
squaredGradientsAvg = [];

% Format the transits with the dimension labels 'SSCB' (spatial, spatial, channel, batch), 
% and the numeric features and targets with the dimension labels 'CB' (channel, batch). 

mbq = minibatchqueue(fdsTrainCombined,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat',{'SSCB','CB','CB'});

%For each epoch, shuffle the data and loop over mini-batches of data. 
%At the end of each epoch, display the training progress. For each mini-batch:
%Evaluate the model gradients, state, and loss using dlfeval and the modelGradients function and update the network state.
%Update the network parameters with Adam, using the adamupdate function.

%Initialize the training progress plot.
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

targs = readall(fdsValCombined.UnderlyingDatastores{3});
% Train the model.
iteration = 0;
start = tic;
% Loop over epochs.
for epoch = 1:numEpochs
    
    % Reduce the learning rate every now and then.
    if epoch == 3
        learnRate = learnRate/2;
    end
    if epoch == 5
        learnRate = learnRate/2;
    end
    if epoch == 8
        learnRate = learnRate/2;
    end
 
    % Shuffle data.
    shuffle(mbq)
        
    % Loop over mini-batches.
    while hasdata(mbq)

        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [dlX1,dlX2,dlY] = next(mbq);
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss] = dlfeval(@modelGradients,dlnet,dlX1,dlX2,dlY);
        dlnet.State = state;
        
        % Update the network parameters using the Adam optimizer.
        [dlnet, gradientsAvg,squaredGradientsAvg] = adamupdate(dlnet, gradients, ...
            gradientsAvg, squaredGradientsAvg, iteration, learnRate, 0.9, 0.95);
        
        ll = double(gather(extractdata(loss)));
        
        if plots == "training-progress"
            % Display the training progress.
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title("Epoch: " + epoch + ", Elapsed: " + string(D));
            addpoints(lineLossTrain,iteration,ll)
            drawnow
        end
        
        % Print loss every 100 iterations.
        if mod(iteration,100) == 0
            ll
        end
    end    
end

save('trainedNet_multi_input_final.mat','dlnet','gradientsAvg','squaredGradientsAvg')

% Evaluate on validation data and print metric score.
disp('Computing score on validation set...')
mbqVal = minibatchqueue(fdsValCombined,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat',{'SSCB','CB','CB'});
predsVal = extractdata(modelPredictions(dlnet,mbqVal));
score = arielMetric(predsVal, targs)



