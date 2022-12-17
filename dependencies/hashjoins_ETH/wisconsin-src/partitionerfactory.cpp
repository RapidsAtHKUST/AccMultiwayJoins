/*
    Copyright 2011, Spyros Blanas.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "partitionerfactory.h"
#include "exceptions.h"

using namespace std;

Partitioner* PartitionerFactory::createPartitioner(const libconfig::Config& cfg, const libconfig::Setting& node)
{
	const libconfig::Setting& hashnode = cfg.getRoot()["partitioner"]["hash"];
	Partitioner* partitioner;
	string partstr = node["algorithm"];
	if (partstr == "no") { 
		partitioner = new Partitioner(cfg, node, hashnode);
	} else if (partstr == "parallel") {
		partitioner = new ParallelPartitioner(cfg, node, hashnode);
	} else if (partstr == "independent") {
		partitioner = new IndependentPartitioner(cfg, node, hashnode);
	} else if (partstr == "derek") {
		partitioner = new DerekPartitioner(cfg, node, hashnode);
	} else if (partstr == "radix") {
		partitioner = new RadixPartitioner(cfg, node, hashnode);
	} else {
		throw UnknownPartitionerException();
	}
	return partitioner;
}
