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
  -- Returns an iterator that iterates from `from` to `to`, inclusive.
  -- From https://stackoverflow.com/a/39690945/6578978
  return coroutine.wrap(function()
    for i = from, to do
      coroutine.yield(i)
    end
  end)
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