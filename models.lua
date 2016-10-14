-- local Squeeze, parent = torch.class('nn.Squeeze', 'nn.Module')

-- function Squeeze:updateOutput(input)
--   self.size = input:size()
--   self.output = input:squeeze()
--   return self.output
-- end

-- function Squeeze:updateGradInput(input, gradOutput)
--   self.gradInput = gradOutput:view(self.size)
--   return self.gradInput
-- end

local Models = {}

-------------------------
-- Convolutional model --
-------------------------

function Models.ConvCuisine(cc, data)
    -- init
    local opti = cc.ingredient
    local input = nn.Identity()()
    local inputSize = opti.size

    -- convolution on each ingredient
    local charTypes = #data.idToChar

    -- convolutions
    local layers = {}
    for i, v in pairs(opti.convs) do
        local maps = v[1]
        local width = v[2]
        local seq = nn.Sequential()
        seq:add(nn.LookupTable(charTypes, opti.embedding))
        seq:add(nn.TemporalConvolution(opti.embedding, maps, width))
        seq:add(nn.ReLU())
        seq:add(nn.TemporalMaxPooling(opti.stringSize - width + 1))
        if opti.dropout > 0 then
            seq:add(nn.Dropout(opti.dropout))
        end
        seq:add(nn.Reshape(maps))
        layers[i] = seq(input)
    end

    -- join / dense layer
    local output
    if #opti.convs > 1 then
        output = nn.JoinTable(2)(layers)
    else
        output = layers[1]
    end
    if opti.ls == 1 then
        local seq = nn.Sequential()
        seq:add(nn.Linear(opti.size, opti.size))
        seq:add(nn.Sigmoid())
        output = seq(output)
    end

    -- dense layers
    local result = {nn.Reshape(inputSize)(output)}
    for i = 1, #cc.hsize do
        local layer = nn.Linear(inputSize, cc.hsize[i])(result[i])
        if opti.bn == 1 then
            layer = nn.BatchNormalization(cc.hsize[i])(layer)
        end
        result[i + 1] = nn.Sigmoid()(layer)
        inputSize = cc.hsize[i]
    end

    -- classification
    local seq = nn.Sequential()
    seq:add(nn.Linear(cc.hsize[#cc.hsize], #data.classes))
    seq:add(nn.LogSoftMax())
    seq:add(nn.Squeeze())
    return nn.gModule({input}, {seq(result[#result])})
end

return Models
