classdef post_data < handle
    
    properties
        N % number of data points
        posts % struct containing all post info including text
        wordList % list of unique words in dataset
        wordCounts % counts of each unique word corresponding to wordList in dataset
        numWords % number of unique words
        tdm % document term matrix, column index corresponds to post index
        
        % individual properties for each post field (more convenient for
        % some calculations)
        id % unique ID for post
        trusted_judgements % number of people who judged post for message
        last_judgement % judgement date
        audience % intended audience (national, constituency)
        audience_confidence % 0.5-1.0 ranking of audience classification confidence
        audience_labels % list of possible audience labels (national, constituency)
        bias % partisan or neutral
        bias_confidence % 0.5-1.0 ranking of bias classification confidence
        bias_labels % list of possible bias labels (partisan, neutral)
        message % classification of message 
        message_confidence % see above
        message_labels % list of possible message labels
        label % info on poster
        source % twitter or facebook
        text % text of post
        parsed_text % text of post parsed into words
    end
    
    methods
        function obj = post_data(filename)
            % read file
            doc = fopen(filename);
            formatSpec = '%u %u %{M/d/yy H:m}D %s %u %s %f %s %f %s %s %q';
            C = textscan(doc,formatSpec,'HeaderLines',1,'Delimiter',{',','\n'});
            obj.N = size(C{1},1);
            fclose(doc);

            % build structure
            obj.posts = cell(obj.N,1);
            obj.id = zeros(obj.N,1);
            obj.trusted_judgements = zeros(obj.N,1);
            obj.last_judgement = cell(obj.N,1);
            obj.audience = cell(obj.N,1);
            obj.audience_confidence = zeros(obj.N,1);
            obj.bias = cell(obj.N,1);
            obj.bias_confidence = zeros(obj.N,1);
            obj.message = cell(obj.N,1);
            obj.message_confidence = zeros(obj.N,1);
            obj.label = cell(obj.N,1);
            obj.source = cell(obj.N,1);
            obj.text = cell(obj.N,1);
            obj.parsed_text = cell(obj.N,1);
            for ii=1:obj.N
                obj.posts{ii} = struct('id',C{1}(ii), ...
                                       'trusted_judgements',C{2}(ii), ...
                                       'last_judgement', C{3}(ii), ...
                                       'audience', C{4}{ii}, ...
                                       'audience_confidence', C{5}(ii), ...
                                       'bias', C{6}{ii}, ...
                                       'bias_confidence', C{7}(ii), ...
                                       'message', C{8}{ii}, ...
                                       'message_confidence', C{9}(ii), ...
                                       'label', C{10}{ii}, ...
                                       'source', C{11}{ii}, ...
                                       'text', C{12}{ii}, ...
                                       'parsed_text',[]);
                obj.id(ii) = obj.posts{ii}.id;
                obj.trusted_judgements(ii) = obj.posts{ii}.trusted_judgements;
                obj.last_judgement{ii} = obj.posts{ii}.last_judgement;
                obj.audience{ii} = obj.posts{ii}.audience;
                obj.audience_confidence(ii) = obj.posts{ii}.audience_confidence;
                obj.bias{ii} = obj.posts{ii}.bias;
                obj.bias_confidence(ii) = obj.posts{ii}.bias_confidence;
                obj.message{ii} = obj.posts{ii}.message;
                obj.message_confidence(ii) = obj.posts{ii}.message_confidence;
                obj.label{ii} = obj.posts{ii}.label;
                obj.source{ii} = obj.posts{ii}.source;
                obj.text{ii} = obj.posts{ii}.text;
                obj.parsed_text{ii} = obj.posts{ii}.parsed_text;
            end
            
            % parse text
            parse_delimiters = {',','.','/','<','>','?',':',';','"','[','{',']','}','\', ...
                                '|','-','_','=','+','`','~','1','2','3','4','5','6','7', ...
                                '8','9','0','!','@','#','$','%','^','&','*','(',')',' ', ...
                                '\\','\0','\a','\b','\f','\n','\r','\t','\v',''''};
                                
            for ii=1:obj.N
                obj.posts{ii}.parsed_text = strsplit(obj.posts{ii}.text,parse_delimiters);
                obj.parsed_text{ii} = obj.posts{ii}.parsed_text;
            end
            
            % identify words
            wordList_u = cell(1,1); wordCounts_u = [];
            % initialize
            wordList_u{1} = obj.parsed_text{1}{1};
            wordCounts_u(1) = 1;
            for ii=1:obj.N
%                 for jj = 1:length(obj.posts{ii}.parsed_text)
                for jj = 1:length(obj.parsed_text{ii})                    
                    % see if word has already been identified
%                     idx = strcmpi(obj.posts{ii}.parsed_text{jj},wordList_u);
                    idx = strcmpi(obj.parsed_text{ii}{jj},wordList_u);
                    if sum(idx)>0
                        % if yes, increment count
                        wordCounts_u(idx) = wordCounts_u(idx)+1;
                    else
                        % if no, add to list
                        wordList_u = [wordList_u; obj.parsed_text{ii}{jj}];
                        wordCounts_u = [wordCounts_u; 1];
                    end
                    if size(wordCounts_u,1) ~= size(wordList_u,1)
                        error('something wrong');
                    end
                end
            end
            
            % sort words by frequency
            [obj.wordCounts,I] = sort(wordCounts_u,'descend');
            obj.wordList = wordList_u(I);
            obj.numWords = length(obj.wordList);
            
            % get labels for all categories
            obj.audience_labels = unique(obj.audience(:));
            obj.bias_labels = unique(obj.bias(:));
            obj.message_labels = unique(obj.message(:));
            
        end
        
        function calc_tdm(obj)
            obj.tdm = zeros(obj.numWords,obj.N);
            
            % loop through each post
            for ii=1:obj.N
                curr_text = obj.parsed_text{ii};
                idxs = zeros(obj.numWords,length(curr_text));
                % loop through parsed text
                for jj=1:length(curr_text)
                    idxs(:,jj) = strcmpi(curr_text{jj},obj.wordList);
                end
                obj.tdm(:,ii) = sum(idxs,2);
            end
        end
        
        function a = tdm_by_message(obj,mes)
            idx = strcmpi(mes,obj.message);
            a = obj.tdm(:,idx);
        end
        function a = tdm_by_audience(obj,mes)
            idx = strcmpi(mes,obj.audience);
            a = obj.tdm(:,idx);
        end
        function a = tdm_by_bias(obj,mes)
            idx = strcmpi(mes,obj.bias);
            a = obj.tdm(:,idx);
        end
    end
    
end

