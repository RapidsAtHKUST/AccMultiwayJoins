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

template <typename Super>
PageCursor* ProbeIsPart<Super>::probe(SplitResult tin, int threadid)
{
	WriteTable* ret = NULL;
	for (int i=threadid; i<tin->size(); i+=Super::nthreads) {
		PageCursor* t = (*tin)[i];
		ret = Super::probeCursor(t, threadid, false, ret);
	}

	return ret;
}

template <typename Super>
PageCursor* ProbeIsNotPart<Super>::probe(SplitResult tin, int threadid)
{
	PageCursor* t = (*tin)[0];
	return Super::probeCursor(t, threadid, true);
}

template <typename Super>
PageCursor* ProbeSteal<Super>::probe(SplitResult tin, int threadid)
{
	WriteTable* ret = NULL;
	for (int i=threadid; i<tin->size(); i+=Super::nthreads) {
		PageCursor* t = (*tin)[i];
		ret = Super::probeCursor(t, threadid, false, ret);
	}

	for (int i=0; i<tin->size(); ++i) {
		PageCursor* t = (*tin)[i];
		ret = Super::probeCursor(t, threadid, true, ret);
	}

	return ret;
}
