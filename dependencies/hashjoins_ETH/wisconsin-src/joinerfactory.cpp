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

#include "joinerfactory.h"
#include "exceptions.h"

using namespace std;

BaseAlgo* JoinerFactory::createJoiner(const libconfig::Config& root) {
	BaseAlgo* joiner;

	const libconfig::Setting& cfg = root.getRoot();
	string flatmem = "no";
	cfg["algorithm"].lookupValue("flatmem", flatmem);
	string copydata = cfg["algorithm"]["copydata"];
	string partitionbuild = cfg["algorithm"]["partitionbuild"];
	string partitionprobe = cfg["algorithm"]["partitionprobe"];
	string steal = "no";
	cfg["algorithm"].lookupValue("steal", steal);

	if (flatmem == "yes") {
		joiner = new FlatMemoryJoiner(root);
	} else if (copydata == "yes") { 
		if (steal == "yes") {
			assert(partitionbuild == "no");
			assert(partitionprobe == "yes");
			joiner = new ProbeSteal < BuildIsNotPart < StoreCopy > > (cfg);
		}
		else if (partitionbuild == "yes") {
			if (partitionprobe == "yes") {
				joiner = new ProbeIsPart< BuildIsPart< StoreCopy > > (cfg);
			} else {
				joiner = new ProbeIsNotPart< BuildIsPart< StoreCopy > > (cfg);
			}
		} else {
			if (partitionprobe == "yes") {
				joiner = new ProbeIsPart< BuildIsNotPart< StoreCopy > > (cfg);
			} else {
				joiner = new ProbeIsNotPart< BuildIsNotPart< StoreCopy > > (cfg);
			}
		}
	} else {
		if (partitionbuild == "yes") {
			if (partitionprobe == "yes") {
				joiner = new ProbeIsPart< BuildIsPart< StorePointer > > (cfg);
			} else {
				joiner = new ProbeIsNotPart< BuildIsPart< StorePointer > > (cfg);
			}
		} else {
			if (partitionprobe == "yes") {
				joiner = new ProbeIsPart< BuildIsNotPart< StorePointer > > (cfg);
			} else {
				joiner = new ProbeIsNotPart< BuildIsNotPart< StorePointer > > (cfg);
			}
		}
	}
	//} else {
	//	throw UnknownAlgorithmException();
	//}
	return joiner;
}
