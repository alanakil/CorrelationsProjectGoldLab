% Compute spike count covariance matrix.
% s is a 2x(ns) matrix where ns is the number of spikes
% s(1,:) lists spike times
% and s(2,:) lists corresponding neuron indices
% Neuron indices are assumed to go from 1 to N
%
% Spikes are counts starting at time T1 and ending at 
% time T2.
%
% winsize is the window size over which spikes are counted,
% so winsize is assumed to be much smaller than T2-T1
%
% Covariances are only computed between neurons whose
% indices are listed in the vector Inds. If Inds is not
% passed in then all NxN covariances are computed.

function C=SpikeCountCovAriana(s,N,T1,T2,winsize,Inds,BlockTrial,TrialTime,...
    T,Control,ControlTrials,LaserTrials, block_slide_size)
    
    % Default: Use all indices
    if(~exist('Inds','var') || isempty(Inds))
        Inds=1:N;
    end

    Inds=round(Inds);
    % Check for out of bounds Inds
%     if(min(Inds)<1 || max(Inds)>N)
%         error('Out of bounds indicesin Inds!');
%     end
        
    % If Control is 1, calculate corrs for control trials. Otherwise,
    % calculate corrs for laser trials.
    i=1;
    if(Control)
        ControlTrialsUsed = ControlTrials(BlockTrial:block_slide_size+BlockTrial-1);
        for t=1:T/TrialTime % T/TrialTime=TrialperBlock
            if(ismember(t,ControlTrialsUsed)) % Count spikes only if t is a control trial.
            
                edgest=T1+(t-1)*TrialTime:winsize:T2+(t-1)*TrialTime; %Count spikes between T1 and T2 for each trial.
                edgesi=Inds;
                edges={edgest,edgesi};
            
                s(2,:)=round(s(2,:));
                s(1,:)=round(s(1,:));

                st=s(:,:);
                % Get 2D histogram of spike indices and trials
                counts=hist3(st','Edges',edges);
            
                % Get rid of edges, which are 0
                counts=counts(1:end-1,1:end);
                % Only use counts that are in Inds.
                counts = counts(:,Inds);
                Counts(i,:) = sum(counts, 1);
                i=i+1;
                clear st counts
            end
        end
        
        % Subtract mean spike count to each neuron.
        Counts = Counts - mean(Counts,1);
        % Compute covariance matrix
        C=cov(Counts);

    else
        LaserTrialsUsed = LaserTrials(BlockTrial:block_slide_size+BlockTrial-1);
        for t=1:T/TrialTime % T/TrialTime=TrialperBlock
            if(ismember(t,LaserTrialsUsed))
            
                edgest=T1+(t-1)*TrialTime:winsize:T2+(t-1)*TrialTime; %Count spikes between T1 and T2 for each trial.
                edgesi=Inds;
                edges={edgest,edgesi};
                s(2,:)=round(s(2,:));
            
                st=s(:,:);
                % Get 2D histogram of spike indices and trials
                counts=hist3(st','Edges',edges);
                
                % Get rid of edges, which are 0
                counts=counts(1:end-1,1:end);

                % Only use counts that are in Inds.
                counts=counts(:,Inds);
                Counts(i,:) = sum(counts, 1);
                i=i+1;
                clear st counts
            end
        end
    
        % Subtract mean spike count to each neuron.
        Counts = Counts - mean(Counts,1);
        % Compute covariance matrix
        C=cov(Counts);
    
    end
    
end
