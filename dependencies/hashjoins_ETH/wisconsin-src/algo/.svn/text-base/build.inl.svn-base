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
void BuildIsPart<Super>::build(SplitResult tin, int threadid)
{
	for (int i=threadid; i<tin->size(); i+=Super::nthreads) {
		PageCursor* t = (*tin)[i];
		Super::buildCursor(t, threadid, false);
	}
}

template <typename Super>
void BuildIsNotPart<Super>::build(SplitResult tin, int threadid)
{
	PageCursor* t = (*tin)[0];
	Super::buildCursor(t, threadid, true);
}
