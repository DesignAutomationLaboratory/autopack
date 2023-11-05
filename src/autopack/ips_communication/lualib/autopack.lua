local module = {}

local base64 = require("base64")
local inspect = require("inspect")
local msgpack = require("MessagePack")

local function pack(data)
  return base64.encode(msgpack.pack(data))
end

local function unpack(string)
  return msgpack.unpack(base64.decode(string))
end

local function ips3VecToTable(vec)
  local table = {}

  for i = 1, 3 do
    -- IPS vectors are 0-indexed
    table[i] = vec[i - 1]
  end

  return table
end

local function ipsNVecToTable(vector)
  local table = {}

  for i = 1, vector:size() do
    -- IPS vectors are 0-indexed
    local element = vector[i - 1]

    -- If the element is a userdata, assume it's a 3-vector
    if type(element) == "userdata" then
      table[i] = ips3VecToTable(element)
    else
      table[i] = element
    end
  end

  return table
end

local function range(from, to)
  -- Returns an array with values from `from` to `to`, inclusive.
  local arr = {}
  for v = from, to do
    arr[#arr + 1] = v
  end
  return arr
end


module.pack = pack
module.unpack = unpack
module.ips3VecToTable = ips3VecToTable
module.ipsNVecToTable = ipsNVecToTable
module.range = range

module.base64 = base64
module.inspect = inspect
module.msgpack = msgpack

return module
