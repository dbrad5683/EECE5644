clear; clc; close all

%%
% read file
docMod = fopen('political_social_media_mod.csv');
formatSpecMod = '%u %u %{M/d/yy H:m}D %s %u %s %f %s %f %s %s %q';
C = textscan(docMod,formatSpecMod,'HeaderLines',1,'Delimiter',{',','\n'});
N = size(C{1},1);
fclose(docMod);

%%
% build structure
postDataset = cell(N,1);
for ii=1:N
    postDataset{ii} = struct('id',C{1}(ii), ...
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
end

%%
% parse text
parse_delimiters = {',','.','/','<','>','?',':',';','"','[','{',']','}','\', ...
                    '|','-','_','=','+','`','~','1','2','3','4','5','6','7', ...
                    '8','9','0','!','@','#','$','%','^','&','*','(',')',' ', ...
                    '\\','\0','\a','\b','\f','\n','\r','\t','\v',''''};
for ii=1:N
    postDataset{ii}.parsed_text = strsplit(postDataset{ii}.text,parse_delimiters);
end

%%
% identify words
wordList = cell(1,1); wordCounts = [];
% initialize
wordList{1} = postDataset{1}.parsed_text{1};
wordCounts(1) = 1;
ignoreList = {'','http','com','and','or','the','a'};
for ii=1:N
    for jj = 1:length(postDataset{ii}.parsed_text)
        % see if word has already been identified
        idx = strcmpi(postDataset{ii}.parsed_text{jj},wordList);
        if sum(idx)>0
            % if yes, increment count
            wordCounts(idx) = wordCounts(idx)+1;
        else
            % if no, add to list
            wordList = [wordList; postDataset{ii}.parsed_text{jj}];
            wordCounts = [wordCounts; 1];
        end
        if size(wordCounts,1) ~= size(wordList,1)
            disp('error');
        end
    end
end

%%
% sort words by frequency

[wordCountsSort,I] = sort(wordCounts,'descend');
wordListSort = wordList(I);
